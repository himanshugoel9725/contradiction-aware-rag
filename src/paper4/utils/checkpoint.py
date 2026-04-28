"""
Checkpoint manager for crash-safe experiment progress tracking.

WHY: Experiments involve hundreds of API calls costing money. If interrupted,
we must resume exactly where we left off — not re-process completed examples.

HOW:
    1. Progress stored as append-only JSONL (one line per completed example)
    2. After each example, result is IMMEDIATELY flushed to disk (crash-safe)
    3. On restart, reads progress file to skip completed examples
    4. _DONE marker file indicates full completion
    5. If _DONE exists, entire experiment is skipped on rerun

USAGE:
    mgr = CheckpointManager("exp_1_1", checkpoint_dir="./checkpoints")

    if mgr.is_complete():
        results = mgr.load_completed()
    else:
        remaining, completed = mgr.start_or_resume(items, id_key="example_id")
        for item in remaining:
            result = process(item)
            mgr.save(item["example_id"], result)
            completed[item["example_id"]] = result
        mgr.finalize(completed)
"""

import json
import os
from pathlib import Path
from typing import Any


class CheckpointManager:
    """
    Manages crash-safe checkpointing for iterative experiment loops.

    Each instance tracks progress for one experiment. Multiple experiments
    can run with separate CheckpointManager instances.

    Attributes:
        experiment_name: Unique name for this experiment (used in filenames)
        checkpoint_dir:  Directory where progress files are stored
        progress_path:   Path to the JSONL progress file
        done_path:       Path to the _DONE marker file
        results_path:    Path to the final complete results JSON
    """

    def __init__(self, experiment_name: str, checkpoint_dir: str = "./checkpoints") -> None:
        """
        Args:
            experiment_name: Unique identifier. Used in filenames; must be filesystem-safe.
            checkpoint_dir:  Where to store checkpoint files. Created if missing.
        """
        self.experiment_name = experiment_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.progress_path = self.checkpoint_dir / f"{experiment_name}_progress.jsonl"
        self.done_path = self.checkpoint_dir / f"{experiment_name}_DONE"
        self.results_path = self.checkpoint_dir / f"{experiment_name}_results.json"

    def is_complete(self) -> bool:
        """True if _DONE marker exists (experiment fully finished in a prior run)."""
        return self.done_path.exists()

    def load_completed(self) -> dict[str, Any]:
        """Load final results dict from a completed experiment."""
        with open(self.results_path, "r") as f:
            return json.load(f)

    def start_or_resume(
        self, items: list[dict], id_key: str = "example_id"
    ) -> tuple[list[dict], dict[str, Any]]:
        """
        Resume from checkpoint or start fresh.

        Reads the progress JSONL (if it exists), builds set of completed IDs,
        returns only the remaining items.

        Args:
            items:  Full list of items to process. Each must have the id_key field.
            id_key: Key in each item dict containing its unique identifier.

        Returns:
            (remaining_items, completed_results_dict)
        """
        completed: dict[str, Any] = {}

        if self.progress_path.exists():
            with open(self.progress_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    completed[record["item_id"]] = record["result"]

        remaining = [item for item in items if item[id_key] not in completed]
        n_total, n_done = len(items), len(completed)

        if n_done > 0:
            print(
                f"[Checkpoint] Resuming '{self.experiment_name}': "
                f"{n_done}/{n_total} done, {len(remaining)} remaining"
            )
        else:
            print(
                f"[Checkpoint] Starting '{self.experiment_name}': {n_total} items"
            )

        return remaining, completed

    def save(self, item_id: str, result: Any) -> None:
        """
        Append one completed example to the progress file.

        Flushes and fsyncs immediately — if the process crashes right after
        this call, the result is still on disk.
        """
        record = {"item_id": item_id, "result": result}
        with open(self.progress_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def finalize(self, all_results: dict[str, Any]) -> None:
        """
        Mark experiment complete. Writes final results JSON and _DONE marker.

        After this, is_complete() returns True and the experiment is skipped on rerun.
        """
        with open(self.results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        self.done_path.touch()
        print(
            f"[Checkpoint] Finalized '{self.experiment_name}': "
            f"{len(all_results)} results saved"
        )

    def get_progress_count(self) -> int:
        """Count completed examples without loading all data."""
        if not self.progress_path.exists():
            return 0
        with open(self.progress_path, "r") as f:
            return sum(1 for line in f if line.strip())

    def reset(self) -> None:
        """Delete all checkpoint files. WARNING: irreversible."""
        for path in [self.progress_path, self.done_path, self.results_path]:
            if path.exists():
                path.unlink()
        print(f"[Checkpoint] Reset '{self.experiment_name}'")


if __name__ == "__main__":
    """Smoke test: verify save/resume/finalize lifecycle."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager("test", checkpoint_dir=tmpdir)
        assert not mgr.is_complete()

        items = [{"example_id": f"ex_{i}", "data": i} for i in range(5)]

        # Start fresh
        remaining, completed = mgr.start_or_resume(items)
        assert len(remaining) == 5 and len(completed) == 0

        # Process first 3 then "crash"
        for item in remaining[:3]:
            mgr.save(item["example_id"], {"score": 0.95})
            completed[item["example_id"]] = {"score": 0.95}

        # Resume — should see 3 done, 2 remaining
        remaining2, completed2 = mgr.start_or_resume(items)
        assert len(remaining2) == 2, f"Expected 2 remaining, got {len(remaining2)}"
        assert len(completed2) == 3, f"Expected 3 completed, got {len(completed2)}"

        # Finish and finalize
        for item in remaining2:
            mgr.save(item["example_id"], {"score": 0.90})
            completed2[item["example_id"]] = {"score": 0.90}

        mgr.finalize(completed2)
        assert mgr.is_complete()
        assert len(mgr.load_completed()) == 5

        print("[PASS] CheckpointManager smoke test passed")

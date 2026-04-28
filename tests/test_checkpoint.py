"""
Test CheckpointManager lifecycle: save, resume, finalize, reset.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.paper4.utils.checkpoint import CheckpointManager


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_fresh_start(tmpdir):
    """Fresh start returns all items and empty completed dict."""
    mgr = CheckpointManager("test", checkpoint_dir=tmpdir)
    items = [{"id": f"ex_{i}"} for i in range(5)]
    remaining, completed = mgr.start_or_resume(items, id_key="id")

    assert len(remaining) == 5
    assert len(completed) == 0
    assert not mgr.is_complete()


def test_save_and_resume(tmpdir):
    """Save 3 items, create new manager, resume → 2 remaining."""
    mgr = CheckpointManager("test", checkpoint_dir=tmpdir)
    items = [{"id": f"ex_{i}"} for i in range(5)]

    remaining, completed = mgr.start_or_resume(items, id_key="id")
    for item in remaining[:3]:
        mgr.save(item["id"], {"score": 0.9})

    # New manager (simulates restart)
    mgr2 = CheckpointManager("test", checkpoint_dir=tmpdir)
    remaining2, completed2 = mgr2.start_or_resume(items, id_key="id")

    assert len(completed2) == 3
    assert len(remaining2) == 2


def test_finalize_and_complete(tmpdir):
    """After finalize, is_complete() returns True."""
    mgr = CheckpointManager("test", checkpoint_dir=tmpdir)
    items = [{"id": "ex_0"}]

    remaining, completed = mgr.start_or_resume(items, id_key="id")
    mgr.save("ex_0", {"score": 0.95})
    completed["ex_0"] = {"score": 0.95}
    mgr.finalize(completed)

    assert mgr.is_complete()
    loaded = mgr.load_completed()
    assert loaded["ex_0"]["score"] == 0.95


def test_reset(tmpdir):
    """Reset clears all checkpoint files."""
    mgr = CheckpointManager("test", checkpoint_dir=tmpdir)
    items = [{"id": "ex_0"}]

    remaining, completed = mgr.start_or_resume(items, id_key="id")
    mgr.save("ex_0", {"score": 0.95})
    completed["ex_0"] = {"score": 0.95}
    mgr.finalize(completed)

    assert mgr.is_complete()

    mgr.reset()
    assert not mgr.is_complete()
    assert mgr.get_progress_count() == 0


def test_progress_count(tmpdir):
    """get_progress_count matches number of saved items."""
    mgr = CheckpointManager("test", checkpoint_dir=tmpdir)
    assert mgr.get_progress_count() == 0

    items = [{"id": f"ex_{i}"} for i in range(10)]
    remaining, _ = mgr.start_or_resume(items, id_key="id")

    for item in remaining[:7]:
        mgr.save(item["id"], {"ok": True})

    assert mgr.get_progress_count() == 7


def test_separate_experiments(tmpdir):
    """Two experiments with different names don't interfere."""
    mgr_a = CheckpointManager("exp_a", checkpoint_dir=tmpdir)
    mgr_b = CheckpointManager("exp_b", checkpoint_dir=tmpdir)

    items = [{"id": "ex_0"}]

    mgr_a.start_or_resume(items, id_key="id")
    mgr_a.save("ex_0", {"result": "a"})

    _, completed_b = mgr_b.start_or_resume(items, id_key="id")
    assert len(completed_b) == 0  # b is independent

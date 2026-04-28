#!/usr/bin/env bash
# Watch /tmp/remaining_phases.log and fire a macOS notification on completion or crash.
LOG=/tmp/remaining_phases.log

echo "[Watcher] Monitoring $LOG ..."

tail -f "$LOG" 2>/dev/null | while IFS= read -r line; do
  if echo "$line" | grep -q "All remaining phases complete"; then
    osascript -e 'display notification "All experiments done! Check figures/ and runs/." with title "contradictionRAG DONE" sound name "Glass"'
    echo "[Watcher] Done notification sent."
    break
  fi
  if echo "$line" | grep -qE "Traceback \(most recent|Error\b|FAILED|set -euo pipefail" && echo "$line" | grep -qvE "^\[Pipeline\]|\[Eval\]|\[BM25\]|\[Embed"; then
    osascript -e "display notification \"Crashed — check /tmp/remaining_phases.log\" with title \"contradictionRAG CRASHED\" sound name \"Basso\""
    echo "[Watcher] Crash notification sent: $line"
    break
  fi
done

#!/bin/bash
# Run this after compare_all_models.py completes to finish Phase 3.
# Usage: bash scripts/finish_phase3.sh

set -e
cd "$(dirname "$0")/.."

VENV=/tmp/ml-vol-momentum-venv/bin/python

echo "=== Running final test suite ==="
$VENV -m pytest tests/ -m "not slow and not shap" -q --tb=short

echo ""
echo "=== Git status ==="
git status

echo ""
echo "=== Tagging phase3-complete ==="
git add -A
git commit -m "chore: Phase 3 complete — 5-model walk-forward, checkpoint gates PASS, surprises updated" || echo "Nothing to commit"
git tag phase3-complete

echo ""
echo "Phase 3 complete. Tag: phase3-complete"
git log --oneline -8

#!/bin/bash
# Push the working tree to Raven. Use ONLY this script -- an ad-hoc rsync has bitten
# this project twice.
#
# What the excludes are protecting against, in the order the incidents happened:
#
#   results/   Raven is where results are PRODUCED. The local repo still carries the
#              ORIGINAL run's results (they are committed), and `run.py` calls
#              os.makedirs(output_dir, exist_ok=False). Sync them over and every
#              evaluation task aborts on contact with its own output directory --
#              after the queue wait. Raven needs nothing from the local results/.
#
#   .env       Exists only on Raven and holds the tokens. A sync with --delete
#              removed it once, and the next job failed with "HF_TOKEN is not set".
#              Never delete it, never overwrite it from here.
#
#   *.key      Three tracked Apple Keynote files. Excluding them makes Raven's
#              `git status` show them as deleted, so a `git commit -a` from Raven
#              would drop them from history. They are excluded because they are
#              large and useless on a cluster -- so never commit from Raven, or
#              pass explicit paths if you must.
#
#   .venv/ dataset/ ckpt/   Built or downloaded on Raven, gigabytes, never travel.
#
# --delete is deliberately NOT used. The risk of removing something Raven owns
# outweighs the tidiness.
#
# Usage:
#   scripts/sync_to_raven.sh              # push
#   scripts/sync_to_raven.sh --dry-run    # show what would change

set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE="${REMOTE:-raven:/u/yinxu/work/Adv-VLM/}"
DRY=()
[[ "${1:-}" == "--dry-run" ]] && DRY=(--dry-run --itemize-changes)

rsync -az --info=stats1 "${DRY[@]}" \
  --exclude='results/' \
  --exclude='.env' \
  --exclude='.venv/' \
  --exclude='dataset/' \
  --exclude='ckpt/' \
  --exclude='PKU-SafeRLHF' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='scripts/logs/' \
  --exclude='*.key' \
  --exclude='report/*.pdf' \
  --exclude='report/*.aux' --exclude='report/*.log' --exclude='report/*.out' \
  --exclude='report/*.fls' --exclude='report/*.fdb_latexmk' --exclude='report/*.blg' \
  --exclude='report/*.bbl' \
  --exclude='.git/' \
  ./ "$REMOTE"

echo ">> pushed to $REMOTE (results/, .env, datasets and checkpoints deliberately untouched)"

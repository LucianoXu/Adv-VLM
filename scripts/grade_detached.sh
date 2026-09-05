#!/bin/bash
# Run a grading config detached from the ssh session.
#
# Grading is a long, network-bound job on a login node: it cannot go through Slurm
# (compute nodes have no internet) and it outlives any reasonable ssh timeout. Every
# inline attempt at `nohup ... &` over ssh either held the channel open until the
# client timed out, or was killed with it. This wraps the pattern once, correctly.
#
# Usage (from the project root on Raven):
#   scripts/grade_detached.sh <config.yaml> <logfile>
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${1:?usage: grade_detached.sh <config.yaml> <logfile>}"
LOG="${2:?usage: grade_detached.sh <config.yaml> <logfile>}"

cat > /tmp/_grade_runner_$$.sh <<INNER
#!/bin/bash
cd "$(pwd)"
module purge >/dev/null 2>&1
module load python-waterboa/2025.06 >/dev/null 2>&1
source .venv/bin/activate
export OMP_NUM_THREADS=1
python -c "from src import run; run('${CFG}')"
echo "GRADE_EXIT=\$?"
INNER
chmod +x /tmp/_grade_runner_$$.sh
setsid nohup /tmp/_grade_runner_$$.sh > "$LOG" 2>&1 < /dev/null &
echo "launched ${CFG} -> ${LOG} (pid $!)"

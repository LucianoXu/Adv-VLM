#!/bin/bash -l
# Generic single-A100 job: run any Adv-VLM experiment config on MPCDF Raven.

#SBATCH --job-name=advvlm
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=125000
#SBATCH --time=08:00:00
#SBATCH --output=scripts/logs/%x_%j.out
#SBATCH --error=scripts/logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p scripts/logs

# Raven provides Python through the module system; our venv is built on it.
module purge 2>/dev/null || true
module load python-waterboa/2025.06
source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false
# Compute nodes have no internet: read pre-downloaded HF assets from cache only.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG="${1:-${CONFIG:-}}"
if [[ -z "${CONFIG}" ]]; then
  echo "ERROR: no config given. Usage: sbatch $0 <path/to/config.yaml>" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config file not found: ${CONFIG}" >&2
  exit 1
fi

echo " >> Job ${SLURM_JOB_ID} on $(hostname)"
echo " >> Config: ${CONFIG}"
echo " >> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

srun python -c "from src import run; run('${CONFIG}')"

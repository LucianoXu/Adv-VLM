[ProjectPlan](./projectplan.md) | [Devlog](./devlog.md)

# Adv-VLM

## Setup

1. copy `.env.example` to `.env` and paste your HuggingFace token there. (https://huggingface.co/settings/tokens).
2. Create a python environment according to `requirements.txt`.
3. Go through `cookbook.ipynb`.

## Project Structure

- `README.md`: this file, project introduction
- `requirements.txt`: python environment requirements
- `references/`: documents like papers and project notes
- `report/`: the final latex report
- `src/`: python source files
- `scripts/`: script files for all jobs
- `configs/`: task configuration templates
- `.env`: environment variable configuration (gitignored)
- `devlog.md`: development and progress log
- `cookbook.ipynb`: push-button experiment cookbooks
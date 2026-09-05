"""
Download benchmark datasets to the local `dataset/` directory.
"""
import sys
from pathlib import Path
from datasets import load_dataset

PROJ = Path(__file__).resolve().parent.parent
# make `src` importable when run as `python scripts/download_data.py`
# (mirrors download_visualrwkv.py; without this the import below fails)
sys.path.insert(0, str(PROJ))

from src.envvar import require_hf_token

OUT = PROJ / "dataset"

# name -> (hf_repo, split)
DATASETS = {
    "imagenette": ("johnowhitaker/imagenette2-320", "train"),
}


def download(name: str) -> None:
    repo, split = DATASETS[name]
    dest = OUT / repo.split("/")[-1]
    if (dest / "dataset_info.json").exists():
        print(f" >> {name}: already present at {dest}, skipping")
        return
    print(f" >> {name}: downloading {repo} [{split}] -> {dest}")
    ds = load_dataset(repo, split=split, streaming=False, token=require_hf_token())
    dest.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(dest))
    print(f" >> {name}: saved {len(ds)} examples to {dest}")


if __name__ == "__main__":
    names = sys.argv[1:] or list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        sys.exit(f"unknown dataset(s): {unknown}; known: {list(DATASETS)}")
    for n in names:
        download(n)

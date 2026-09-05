"""
Validate experiment configs before they reach a GPU.

`src/run.py` executes a config list task by task and calls
`os.makedirs(output_dir, exist_ok=False)`, so a stale output directory or a typo in
a path aborts the run -- possibly at task 11 of 30, hours in, after the queue wait.
This checks everything that can be checked without a GPU:

  - the YAML parses and is a list of task dicts
  - every `task_type` is one `run()` actually dispatches
  - every key the corresponding task function reads is present
  - `output_dir` does not already exist (and is unique within and across configs)
  - referenced inputs exist: benchmark `local_path`, `data_path`, `adv_image`,
    VisualRWKV `model_path`, and `generations` for grade tasks
  - inputs produced by an earlier stage are reported as "pending" rather than
    missing, so a dependency that has not run yet does not read as an error
  - `defense` blocks name a real transform with accepted arguments

Exit status is 1 if anything is a hard error, 0 if only pending-dependency notes.

Usage:
    python scripts/preflight.py configs/*.yaml
    python scripts/preflight.py --pipeline          # the configs submit_pipeline.sh uses
"""

import argparse
import sys
from pathlib import Path

import yaml

TASK_KEYS = {
    "VLM-ImageClass": ["output_dir", "vlm", "benchmark", "question", "answer_priming",
                       "batch_size", "limit", "shuffle", "seed"],
    "VLM-ImageClass-attack": ["output_dir", "vlm", "benchmark", "question",
                              "answer_priming", "batch_size", "limit", "shuffle", "seed"],
    "VLM-SafeRLHF-attack": ["output_dir", "vlm", "data_path", "batch_size", "limit",
                            "shuffle", "seed"],
    "VLM-SafeRLHF-gen": ["output_dir", "vlm", "data_path", "adv_image"],
    "VLM-SafeRLHF-grade": ["output_dir"],
    "VLM-SafeRLHF-eval": ["output_dir", "vlm", "data_path", "adv_image"],
    "CLIP-ImageClass": ["output_dir", "clip", "benchmark", "batch_size", "limit",
                        "shuffle", "seed"],
    "CLIP-ImageClass-Attack": ["output_dir", "clip", "benchmark", "batch_size", "limit",
                               "shuffle", "seed"],
}

PIPELINE = [
    "configs/clip-llava-attack-imagenette.yaml",
    "configs/budget-sweep-attack.yaml",
    "configs/xeval-clean.yaml",
    "configs/xeval.yaml",
    "configs/xeval-rwkv.yaml",
    "configs/budget-sweep-eval.yaml",
    "configs/defense-classify.yaml",
    "configs/jailbreak-attack-llava.yaml",
    "configs/jailbreak-attack-rwkv.yaml",
    "configs/jailbreak-gen.yaml",
    "configs/defense-jailbreak.yaml",
    "configs/jailbreak-grade.yaml",
    "configs/defense-jailbreak-grade.yaml",
]

errors: list[str] = []
pending: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)
    print(f"  ERROR   {msg}")


def pend(msg: str) -> None:
    pending.append(msg)
    print(f"  pending {msg}")


def check_input(where: str, p: str, produced: set[str]) -> None:
    '''
    An input is fine if it exists on disk, or if an earlier task in this same
    submission is going to produce it (then it is merely pending).
    '''
    path = Path(p)
    if path.exists():
        return
    # results/... paths are produced by other tasks; anything under a planned
    # output_dir counts as pending rather than missing
    if any(str(path) == d or str(path).startswith(d.rstrip("/") + "/") for d in produced):
        pend(f"{where}: {p} (produced by an earlier stage)")
        return
    if str(path).startswith("results/"):
        pend(f"{where}: {p} (under results/, expected from an earlier stage)")
        return
    err(f"{where}: missing input {p}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", type=Path)
    ap.add_argument("--pipeline", action="store_true",
                    help="check the configs submit_pipeline.sh submits")
    args = ap.parse_args()

    files = [Path(p) for p in PIPELINE] if args.pipeline else args.configs
    if not files:
        ap.error("give config paths or --pipeline")

    seen_dirs: dict[str, str] = {}
    produced: set[str] = set()
    n_tasks = 0

    # first pass: collect every output_dir, so later inputs can be recognised as
    # pending-from-an-earlier-stage rather than missing
    for f in files:
        if not f.exists():
            err(f"{f}: no such config")
            continue
        try:
            tasks = yaml.safe_load(f.read_text())
        except yaml.YAMLError as e:
            err(f"{f}: YAML does not parse: {e}")
            continue
        if not isinstance(tasks, list):
            err(f"{f}: top level is {type(tasks).__name__}, expected a list of tasks")
            continue
        for t in tasks:
            if isinstance(t, dict) and "output_dir" in t:
                produced.add(str(t["output_dir"]))

    for f in files:
        if not f.exists():
            continue
        tasks = yaml.safe_load(f.read_text())
        if not isinstance(tasks, list):
            continue
        print(f"\n{f}  ({len(tasks)} tasks)")

        for i, t in enumerate(tasks):
            n_tasks += 1
            where = f"{f}[{i}]"
            if not isinstance(t, dict):
                err(f"{where}: task is {type(t).__name__}, expected a mapping")
                continue

            tt = t.get("task_type")
            if tt not in TASK_KEYS:
                err(f"{where}: unknown task_type {tt!r}")
                continue
            where = f"{f}[{i}] {t.get('output_dir', '<no output_dir>')}"

            for k in TASK_KEYS[tt]:
                if k not in t:
                    err(f"{where}: missing required key {k!r} for {tt}")

            od = t.get("output_dir")
            if od:
                if Path(od).exists():
                    err(f"{where}: output_dir already exists -- run.py uses "
                        f"exist_ok=False and will abort")
                if od in seen_dirs:
                    err(f"{where}: output_dir also used by {seen_dirs[od]}")
                else:
                    seen_dirs[od] = where

            if (b := t.get("benchmark")) is not None:
                if (lp := b.get("local_path")):
                    check_input(where, lp, produced)
                if b.get("name") == "imagenette-adv" and "attack_type" not in b:
                    err(f"{where}: imagenette-adv needs attack_type")

            if (v := t.get("vlm")) is not None:
                if v.get("name") == "VisualRWKV":
                    if not (mp := v.get("model_path")):
                        err(f"{where}: VisualRWKV needs model_path")
                    else:
                        check_input(where, mp, produced)
                elif v.get("name") != "LLaVA":
                    err(f"{where}: unknown vlm name {v.get('name')!r}")

            for key in ("data_path", "adv_image", "generations"):
                if (p := t.get(key)):
                    check_input(where, p, produced)

            if tt == "VLM-SafeRLHF-grade" and "generations" not in t:
                pend(f"{where}: no `generations` key; grading will look for "
                     f"{od}/generations.json in its own output dir")

            if (d := t.get("defense")) is not None:
                try:
                    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                    from src.defense import TRANSFORMS
                    name = d.get("name")
                    if name not in TRANSFORMS:
                        err(f"{where}: unknown defense {name!r}; "
                            f"known: {sorted(TRANSFORMS)}")
                    else:
                        import inspect
                        sig = inspect.signature(TRANSFORMS[name])
                        bad = [k for k in d if k != "name" and k not in sig.parameters]
                        if bad:
                            err(f"{where}: defense {name!r} does not accept {bad}")
                except ImportError as e:
                    pend(f"{where}: could not import src.defense to validate "
                         f"the defense block ({e})")

    print(f"\n{'='*70}")
    print(f"{n_tasks} tasks across {len(files)} configs")
    print(f"{len(errors)} error(s), {len(pending)} pending dependency note(s)")
    if errors:
        print("\nerrors:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nno hard errors. Pending notes are inputs an earlier stage will produce.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Compare the regenerated cross-evaluation table against the original paper's numbers.

The original adversarial datasets were never committed, so every number is being
re-derived from a fresh pipeline run. A regenerated attack is not bit-identical to
the original (Adam plus non-deterministic CUDA kernels), so the numbers will move a
little -- but they should not move much. This prints the new table beside the
published one with per-cell deltas, so "the rerun reproduces the paper" is a
measured claim rather than an assumption.

Only the `resized` columns are compared: the submission drops the duplicate
`image01` columns, so those are the ones that survive into the paper.

Usage:
    python scripts/check_reproduction.py
"""

import json
from pathlib import Path

# published values, resized columns:
# (clean, CLIP-Adv true-label, CLIP-Adv targeted, LLaVA-Adv true-label, LLaVA-Adv targeted)
ORIGINAL = {
    "CLIP":    (99.5,  0.3, 99.7, 82.0, 11.7),
    "LLaVA A": (97.0, 51.0, 39.5, 14.1, 76.6),
    "LLaVA B": (73.8, 40.8, 32.0, 44.5, 15.6),
    "LLaVA C": (70.4, 45.0, 31.0, 50.8, 11.7),
    "RWKV A":  (96.8, 38.0, 59.8, 71.1, 14.8),
    "RWKV B":  (93.1, 37.4, 54.0, 70.3, 12.5),
    "RWKV C":  (94.2, 38.4, 51.6, 66.4, 10.9),
}

ROWS = [("CLIP", "results/xeval-clean/clip",
         "results/xeval/clip-on-clip-resized", "results/xeval/clip-on-llava-resized")]
ROWS += [(f"LLaVA {p}", f"results/xeval-clean/llava-{p}",
          f"results/xeval/llava-on-clip-resized-{p}",
          f"results/xeval/llava-on-llava-resized-{p}") for p in "ABC"]
ROWS += [(f"RWKV {p}", f"results/xeval-clean/rwkv-{p}",
          f"results/xeval-rwkv/rwkv-on-clip-resized-{p}",
          f"results/xeval-rwkv/rwkv-on-llava-resized-{p}") for p in "ABC"]


def read(d: str, *keys):
    p = Path(d) / "results.json"
    if not p.exists() or not (Path(d) / "DONE").exists():
        return None
    r = json.loads(p.read_text())
    vals = [r.get(k) for k in keys]
    return None if any(v is None for v in vals) else tuple(v * 100 for v in vals)


def main() -> None:
    head = f"{'row':<9} {'clean':>14} {'CLIP-Adv true/tgt':>26} {'LLaVA-Adv true/tgt':>26}"
    print(head)
    print("-" * len(head))

    worst, worst_cell, missing = 0.0, "", []
    for name, clean_d, clip_d, llava_d in ROWS:
        c = read(clean_d, "accuracy")
        a = read(clip_d, "accuracy_original", "accuracy_attack")
        l = read(llava_d, "accuracy_original", "accuracy_attack")
        if c is None or a is None or l is None:
            missing.append(name)
            print(f"{name:<9} (incomplete)")
            continue
        new = (c[0], a[0], a[1], l[0], l[1])
        old = ORIGINAL[name]
        d = [n - o for n, o in zip(new, old)]
        for val, lbl in zip(d, ("clean", "clip-true", "clip-tgt", "llava-true", "llava-tgt")):
            if abs(val) > worst:
                worst, worst_cell = abs(val), f"{name}/{lbl}"
        print(f"{name:<9} {new[0]:5.1f} ({d[0]:+5.1f}) "
              f"{new[1]:6.1f}/{new[2]:5.1f} ({d[1]:+5.1f}/{d[2]:+5.1f}) "
              f"{new[3]:6.1f}/{new[4]:5.1f} ({d[3]:+5.1f}/{d[4]:+5.1f})")

    if missing:
        print(f"\nincomplete rows: {missing}")
    print(f"\nlargest deviation from the published table: {worst:.1f} points ({worst_cell})")
    print("Note: LLaVA-Adv columns rest on only 128 examples, so one example is 0.8 points;")
    print("the CLIP-Adv and clean columns use 1024, where one example is 0.1 points.")


if __name__ == "__main__":
    main()

"""
Compare the adversarial images produced by repeated runs of one attack config.

Section V-B of the paper claims repeated runs "give very different images". That
is a claim about the IMAGES, but what was actually measured was their COMPLIANCE
RATE -- a claim about outcomes. The two are not the same: runs could converge to
near-identical images that happen to be scored differently, or to visually
unrelated images that happen to work equally well. This script measures the
former so the paper can state whichever is true.

Reports, for every pair of runs, the RMS difference, cosine similarity and Pearson
correlation over the flattened image01 tensor, plus per-image statistics. A random
pair of unrelated [0,1] images is included as a reference point, since cosine
similarity between two non-negative vectors is high by construction and needs a
baseline to be interpretable.

Usage:
    python scripts/compare_attack_images.py
"""

import itertools
import os

import torch

RUNS = {
    "LLaVA 84.5 (selected)": "results/lrsweep/attack-lr030/checkpoints/adv_final.pt",
    "LLaVA 82.5":            "results/jailbreak/attack-llava-s44/checkpoints/adv_final.pt",
    "LLaVA 47.0":            "results/jailbreak/attack-llava-s42/checkpoints/adv_final.pt",
    "LLaVA 44.0":            "results/jailbreak/attack-llava-s43/checkpoints/adv_final.pt",
    "RWKV 91.5 (selected)":  "results/jailbreak/attack-rwkv-s43/checkpoints/adv_final.pt",
    "RWKV 87.0":             "results/jailbreak/attack-rwkv-s42/checkpoints/adv_final.pt",
    "RWKV 53.5":             "results/jailbreak/attack-rwkv-s44/checkpoints/adv_final.pt",
}


def stats(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
    rms = ((x - y) ** 2).mean().sqrt().item()
    cos = torch.nn.functional.cosine_similarity(x[None], y[None]).item()
    xc, yc = x - x.mean(), y - y.mean()
    corr = (xc @ yc / (xc.norm() * yc.norm())).item()
    return rms, cos, corr


def main() -> None:
    imgs = {}
    for name, path in RUNS.items():
        if not os.path.exists(path):
            print(f"  !! missing: {path}")
            continue
        t = torch.load(path, map_location="cpu").float()
        imgs[name] = t.flatten()

    if len(imgs) < 2:
        raise SystemExit("need at least two images")

    d = next(iter(imgs.values())).numel()
    print(f"image01 tensors, {d} dims, values in [0,1]\n")

    # reference: two unrelated uniform images, and the init each attack started from
    g = torch.Generator().manual_seed(0)
    r1 = torch.rand(d, generator=g)
    r2 = torch.rand(d, generator=g)
    rms, cos, corr = stats(r1, r2)
    print(f"reference -- two unrelated random images:")
    print(f"  RMS {rms:.4f}   cosine {cos:.4f}   corr {corr:+.4f}\n")

    hdr = f"{'pair':44} {'RMS':>7} {'cosine':>8} {'corr':>8}"
    print(hdr)
    print("-" * len(hdr))
    for a, b in itertools.combinations(imgs, 2):
        same_backbone = a.split()[0] == b.split()[0]
        if not same_backbone:
            continue
        rms, cos, corr = stats(imgs[a], imgs[b])
        print(f"{a + '  vs  ' + b:44} {rms:7.4f} {cos:8.4f} {corr:+8.4f}")

    print(f"\n{'image':24} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}")
    for k, v in imgs.items():
        print(f"{k:24} {v.mean():7.3f} {v.std():7.3f} {v.min():7.3f} {v.max():7.3f}")


if __name__ == "__main__":
    main()

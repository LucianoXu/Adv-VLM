"""
Regenerate the report figures from the current results.

This exists because the figures in report/figures/ were rendered from the original
adversarial datasets, and those were never committed (dataset/ is gitignored) --
they are gone. Every number in the paper is being re-derived from a fresh pipeline
run, so the figures have to come from that same run, or the figure and the table
would be describing two different sets of adversarial images.

Two figure groups:

  --samples    the clean/adversarial sample panel (fig:dataset).
               Picks the SAME four Imagenette images as the original figure, read
               from report/figures/dataset_figure_info.json, so the paper keeps its
               familiar picture while the pixels come from the new attack. The
               dataset stores each example's Imagenette `index`, and the attack is
               deterministic given (limit, shuffle, seed), so these indices should
               still be present; any that are missing are reported and replaced from
               the front of the dataset.

  --jailbreak  the universal jailbreak panel (fig:saferlhf): the random initial
               image, the final adversarial image, and the nearest-neighbour zoom of
               the central patch that shows the 14x14 grid artefact. The crop
               rectangle matches the one drawn in the LaTeX figure -- see CROP below.

Needs torch, datasets and PIL, but no GPU: run it on a login node.

Usage:
    python scripts/make_figures.py --samples results/clip-imagenette-attack-resized/dataset
    python scripts/make_figures.py --jailbreak results/jailbreak/attack-llava-s42
    python scripts/make_figures.py --samples <ds> --jailbreak <dir>   # both
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

FIG_DIR = Path("report/figures")

# The LaTeX figure draws its crop box in axis-fraction coordinates measured from the
# image's bottom-left corner:  x in [0.1667, 0.3333], y in [0.5833, 0.75].
# Pillow crops from the top-left, so y flips: y_top = 1 - y_upper.
CROP = {"x0": 0.1667, "x1": 0.3333, "y_bottom": 0.5833, "y_top": 0.75}
ZOOM_OUT_PX = 336   # nearest-neighbour upscale target, so single pixels stay visible


def to_pil(t: torch.Tensor) -> Image.Image:
    '''image01 (C,H,W) or (1,C,H,W) -> 8-bit PIL, matching src.image.image012resized.'''
    if t.dim() == 4:
        t = t[0]
    a = (t.detach().cpu().float() * 255).clamp(0, 255).round().to(torch.uint8)
    return Image.fromarray(a.permute(1, 2, 0).numpy())


def as_image01(x) -> torch.Tensor:
    '''
    Dataset rows hold either a PIL image (resized-space attacks, stored lossless as
    PNG) or a raw float array (image01-space attacks, which must stay float or
    rounding to uint8 would erase the perturbation). Normalize both to image01.
    '''
    if isinstance(x, Image.Image):
        import numpy as np
        arr = np.array(x.convert("RGB"))
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    t = torch.as_tensor(x).float()
    return t if t.dim() == 3 else t[0]


def make_samples(ds_path: Path) -> None:
    from datasets import load_from_disk

    ds = load_from_disk(str(ds_path))
    labels = ds.features["original_label"].names
    print(f" >> {ds_path}: {len(ds)} examples, {len(labels)} classes")

    wanted, info_path = [], FIG_DIR / "dataset_figure_info.json"
    if info_path.exists():
        wanted = [e["index"] for e in json.loads(info_path.read_text())]
        print(f" >> reusing the original figure's Imagenette indices: {wanted}")

    # dataset index -> row number
    row_of = {int(v): i for i, v in enumerate(ds["index"])}
    rows, missing = [], []
    for idx in wanted:
        if idx in row_of:
            rows.append(row_of[idx])
        else:
            missing.append(idx)
    if missing:
        print(f" !! Imagenette indices {missing} are not in this dataset "
              f"(different limit/shuffle/seed?); filling from the front instead")
    for i in range(len(ds)):
        if len(rows) >= 4:
            break
        if i not in rows:
            rows.append(i)
    rows = rows[:4]

    out_info = []
    for col, row in enumerate(rows):
        rec = ds[row]
        orig = as_image01(rec["original_image"])
        adv = as_image01(rec["adversarial_image"])
        delta = adv - orig

        to_pil(orig).save(FIG_DIR / f"sample{col}_orig.png")
        to_pil(adv).save(FIG_DIR / f"sample{col}_adv.png")

        entry = {
            "col": col,
            "index": int(rec["index"]),
            "original_label": labels[rec["original_label"]],
            "attack_label": labels[rec["attack_label"]],
            "l2": round(float(delta.norm()), 2),
            "mse": round(float(delta.square().mean()), 7),
            "rms": round(float(delta.square().mean().sqrt()), 5),
            "linf": round(float(delta.abs().max()), 4),
            "size": list(orig.shape[-2:]),
            "source": str(ds_path),
        }
        out_info.append(entry)
        print(f"    col {col}: idx {entry['index']:>6}  "
              f"{entry['original_label']} -> {entry['attack_label']}  "
              f"rms {entry['rms']:.5f}  linf {entry['linf']:.4f}")

    (FIG_DIR / "dataset_figure_info.json").write_text(json.dumps(out_info, indent=2))
    print(f" >> wrote sample0..3_{{orig,adv}}.png and dataset_figure_info.json")
    print(" >> NOTE: the RMS values printed above go in the figure caption; "
          "check them against the caption in the tex.")


def make_jailbreak(attack_dir: Path) -> None:
    ck = attack_dir / "checkpoints"
    init_pt, final_pt = ck / "adv_init.pt", ck / "adv_final.pt"
    for p in (init_pt, final_pt):
        if not p.exists():
            raise SystemExit(f"missing {p}; did the attack run with save_dir set?")

    init = torch.load(init_pt)
    final = torch.load(final_pt)
    to_pil(init).save(FIG_DIR / "saferlhf_init.png")
    adv_img = to_pil(final)
    adv_img.save(FIG_DIR / "saferlhf_adv.png")

    W, H = adv_img.size
    box = (
        int(round(CROP["x0"] * W)),
        int(round((1 - CROP["y_top"]) * H)),        # LaTeX y is measured from the bottom
        int(round(CROP["x1"] * W)),
        int(round((1 - CROP["y_bottom"]) * H)),
    )
    crop = adv_img.crop(box)
    crop.resize((ZOOM_OUT_PX, ZOOM_OUT_PX), Image.NEAREST).save(FIG_DIR / "saferlhf_zoom.png")

    d = (final.float() - init.float())
    print(f" >> {attack_dir}")
    print(f"    perturbation vs init: rms {float(d.square().mean().sqrt()):.4f}  "
          f"linf {float(d.abs().max()):.4f}")
    print(f"    crop box (left, top, right, bottom) = {box}, "
          f"{box[2]-box[0]}x{box[3]-box[1]} px upscaled to {ZOOM_OUT_PX}px")
    print(" >> wrote saferlhf_{init,adv,zoom}.png")
    print(" >> NOTE: the crop box must stay in sync with the tikz rectangle in the "
          "LaTeX figure; CROP at the top of this file documents the shared numbers.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path,
                    help="adversarial dataset dir, e.g. "
                         "results/clip-imagenette-attack-resized/dataset")
    ap.add_argument("--jailbreak", type=Path,
                    help="jailbreak attack output dir, e.g. "
                         "results/jailbreak/attack-llava-s42")
    args = ap.parse_args()
    if not args.samples and not args.jailbreak:
        ap.error("give --samples and/or --jailbreak")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.samples:
        make_samples(args.samples)
    if args.jailbreak:
        make_jailbreak(args.jailbreak)


if __name__ == "__main__":
    main()

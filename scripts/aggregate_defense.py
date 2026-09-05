"""
Aggregate the input-transformation defence sweep into the paper's defence table.

Reads two families of results:

  results/defense/<model>-<cond>-<def>/results.json        classification
  results/defense/grade-jb-llava-<def>-<judge>/results.json  jailbreak

and emits `defense.tex` (a table body) plus `defense.json`.

Two things worth knowing before reading the output:

  The `none` row comes from this sweep, not from the main cross-evaluation table.
  The sweep evaluates 256 images where the main table uses 1024, so the two clean
  accuracies will differ slightly. Taking the baseline from the sweep keeps every
  row in the table comparable to every other row, which matters more here than
  agreeing to the decimal with a different table.

  The `none` row is also a consistency check: it should reproduce the eps030 point
  of the budget sweep, since both evaluate the same 256-image eps=0.03 dataset
  undefended. If it does not, something is wrong with the defence wrapper and the
  whole table is suspect -- the script checks this and says so.

Usage:
    python scripts/aggregate_defense.py [--results-dir results/defense]
                                        [--budget-dir results/budget]
                                        [--out results/defense/aggregate]
"""

import argparse
import json
import re
from pathlib import Path

CLS_RE = re.compile(r"^(?P<model>clip|llava|rwkv)-(?P<cond>clean|adv)-(?P<def>\w+)$")
JB_RE = re.compile(r"^grade-jb-llava-(?P<def>\w+)-(?P<judge>[\w.]+)$")

# row order and display names; `none` first so it reads as the baseline
DEFENSES = [
    ("none",   "None (baseline)"),
    ("jpeg75", "JPEG $q{=}75$"),
    ("blur10", "Gaussian blur $\\sigma{=}1$"),
    ("bits3",  "Bit depth $\\to 3$"),
    ("rspad",  "Random resize-pad"),
]
MODELS = [("llava", "LLaVA"), ("rwkv", "VisualRWKV"), ("clip", "CLIP")]


def read(d: Path) -> dict | None:
    if not (d / "DONE").exists() or not (d / "results.json").exists():
        return None
    return json.loads((d / "results.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/defense", type=Path)
    ap.add_argument("--budget-dir", default="results/budget", type=Path)
    ap.add_argument("--out", default="results/defense/aggregate", type=Path)
    args = ap.parse_args()

    cls: dict[tuple[str, str, str], dict] = {}
    jb: dict[str, dict] = {}
    for d in sorted(args.results_dir.glob("*")):
        if not d.is_dir():
            continue
        if (m := CLS_RE.match(d.name)):
            if (r := read(d)) is not None:
                cls[(m.group("model"), m.group("cond"), m.group("def"))] = r
            else:
                print(f" !! incomplete: {d.name}")
        elif (m := JB_RE.match(d.name)):
            if (r := read(d)) is not None:
                jb[m.group("def")] = r
            else:
                print(f" !! incomplete: {d.name}")

    print(f" >> {len(cls)} classification cells, {len(jb)} graded jailbreak cells")

    rows = []
    for dtag, label in DEFENSES:
        row = {"defense": dtag, "label": label}
        for mdl, _ in MODELS:
            c = cls.get((mdl, "clean", dtag))
            a = cls.get((mdl, "adv", dtag))
            row[f"{mdl}_clean"] = c.get("accuracy") if c else None
            row[f"{mdl}_robust"] = a.get("accuracy_original") if a else None
            row[f"{mdl}_targeted"] = a.get("accuracy_attack") if a else None
        g = jb.get(dtag)
        row["jb_adv"] = g.get("adversarial_compliance_rate") if g else None
        row["jb_clean"] = g.get("clean_compliance_rate") if g else None
        rows.append(row)

    # consistency check against the budget sweep's undefended eps030 point
    note = None
    base = next(r for r in rows if r["defense"] == "none")
    for mdl, name in MODELS:
        bres = read(args.budget_dir / f"eval-{mdl}-eps030")
        got, want = base[f"{mdl}_targeted"], (bres or {}).get("accuracy_attack")
        if got is None or want is None:
            continue
        delta = abs(got - want)
        status = "OK" if delta < 1e-6 else f"MISMATCH by {delta:.4f}"
        line = (f"    {name}: defence `none` targeted={got:.4f} vs "
                f"budget eps030 targeted={want:.4f}  -> {status}")
        if delta >= 1e-6:
            note = "mismatch"
        print(line)
    if note:
        print(" !! the undefended row does not reproduce the budget sweep. The two "
              "evaluate the same dataset with no transformation, so they must agree "
              "exactly. Check the defence wrapper before using this table.")
    elif any(base[f"{m}_targeted"] is not None for m, _ in MODELS):
        print(" >> undefended row reproduces the budget sweep exactly")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "defense.json").write_text(json.dumps(rows, indent=2))

    def pct(x, nd=1):
        return f"{x*100:.{nd}f}" if x is not None else "\\PH{--}"

    # compact form: LLaVA classification + the LLaVA jailbreak image, matching the
    # column layout already reserved in the tex
    tex = [
        "% generated by scripts/aggregate_defense.py -- do not edit by hand",
        "% Imagenette: LLaVA prompt A, 256 images, eps=0.03 encoder-level attack.",
        "% SafeRLHF: LLaVA white-box jailbreak image (seed 42), 500 held-out prompts.",
        "% The `none` row is this sweep's own baseline, not the 1024-image main table.",
        "\\begin{tabular}{@{}l cc cc@{}}",
        "\\toprule",
        "\\multirow{2}{*}{Transformation} & \\multicolumn{2}{c}{Imagenette} "
        "& \\multicolumn{2}{c}{SafeRLHF} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "  & clean & targeted & clean & adv. \\\\",
        "\\midrule",
    ]
    for r in rows:
        tex.append(f"{r['label']} & {pct(r['llava_clean'])} & {pct(r['llava_targeted'])} "
                   f"& {pct(r['jb_clean'], 0)} & {pct(r['jb_adv'], 0)} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (args.out / "defense.tex").write_text("\n".join(tex) + "\n")

    # wide form: all three models, in case the page budget allows it
    wide = [
        "% generated by scripts/aggregate_defense.py -- wide variant, all three models",
        "\\begin{tabular}{@{}l cc cc cc cc@{}}",
        "\\toprule",
        "\\multirow{2}{*}{Transformation} & \\multicolumn{2}{c}{CLIP} "
        "& \\multicolumn{2}{c}{LLaVA} & \\multicolumn{2}{c}{VisualRWKV} "
        "& \\multicolumn{2}{c}{SafeRLHF} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
        "  & clean & tgt. & clean & tgt. & clean & tgt. & clean & adv. \\\\",
        "\\midrule",
    ]
    for r in rows:
        wide.append(
            f"{r['label']} & {pct(r['clip_clean'])} & {pct(r['clip_targeted'])} "
            f"& {pct(r['llava_clean'])} & {pct(r['llava_targeted'])} "
            f"& {pct(r['rwkv_clean'])} & {pct(r['rwkv_targeted'])} "
            f"& {pct(r['jb_clean'], 0)} & {pct(r['jb_adv'], 0)} \\\\")
    wide += ["\\bottomrule", "\\end{tabular}"]
    (args.out / "defense-wide.tex").write_text("\n".join(wide) + "\n")

    hdr = (f"{'transformation':<26} " +
           " ".join(f"{n+' clean':>13} {n+' tgt':>11}" for _, n in MODELS) +
           f"{'jb clean':>10} {'jb adv':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = "".join(
            f"{pct(r[f'{m}_clean']):>13} {pct(r[f'{m}_targeted']):>11} " for m, _ in MODELS)
        print(f"{r['defense']:<26} {cells}{pct(r['jb_clean'],0):>10} {pct(r['jb_adv'],0):>8}")

    print(f"\n >> wrote {args.out}/defense.{{json,tex}} and defense-wide.tex")


if __name__ == "__main__":
    main()

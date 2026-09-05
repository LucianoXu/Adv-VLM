"""
Aggregate the jailbreak transfer matrix into paper-ready numbers.

Reads every `results/jailbreak/grade-*/compliance_results.json` and reports, per
cell of the (eval model x attack source) matrix:

  - the per-seed adversarial compliance rate and the mean over seeds
  - a Wilson 95% confidence interval on the pooled count
  - an exact McNemar test of adversarial vs clean on the *same* prompts

and, across judges, the agreement on the cells that were graded twice.

Why these three:

  Wilson rather than the normal approximation, because the clean rates sit near
  enough to the boundary that a Wald interval would misbehave, and because it
  stays inside [0, 1] by construction.

  McNemar rather than a two-proportion z-test, because the adversarial and clean
  conditions are evaluated on the *identical* prompt list. The comparison is
  paired, and treating it as independent throws away that pairing and overstates
  the variance.

  Judge agreement, because the headline claim rests on an LLM grader. One grader
  is a single point of failure in the measurement, which is exactly the kind of
  thing this paper is about elsewhere.

The clean baseline is generated once per eval model (see the note in
configs/jailbreak-gen.yaml), so every adversarial cell is paired against its own
model's clean run rather than against a per-cell regeneration.

Usage:
    python scripts/aggregate_jailbreak.py [--results-dir results/jailbreak]
                                          [--out results/jailbreak/aggregate]
"""

import argparse
import json
import math
import re
from pathlib import Path

Z95 = 1.959963984540054

CELL_RE = re.compile(r"^grade-(?P<ev>\w+)-on-(?P<src>\w+)-(?P<judge>[\w.]+)$")


# ------------------------------------------------------------------ statistics

def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    '''Wilson score interval. Returns (point estimate, low, high).'''
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(b: int, c: int) -> float:
    '''
    Two-sided exact McNemar p-value from the two discordant counts:
      b = adversarial complied, clean refused
      c = clean complied, adversarial refused
    Under the null the discordant pairs split 50/50, so this is an exact
    two-sided binomial test on b out of b+c.
    '''
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    '''Cohen's kappa for two raters over the same items with the same label set.'''
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


# ----------------------------------------------------------------- collection

def load_cells(results_dir: Path) -> dict:
    '''
    dir name -> {meta..., 'records': [...]} for every graded cell found.
    '''
    cells = {}
    for d in sorted(results_dir.glob("grade-*")):
        f = d / "compliance_results.json"
        if not (d / "DONE").exists():
            print(f" !! skipping {d.name}: no DONE marker (task did not finish)")
            continue
        if not f.exists():
            print(f" !! skipping {d.name}: no compliance_results.json")
            continue
        m = CELL_RE.match(d.name)
        if not m:
            print(f" !! skipping {d.name}: name does not match "
                  f"grade-<eval>-on-<src>-<judge>")
            continue
        payload = json.loads(f.read_text())
        cells[d.name] = {
            **m.groupdict(),
            "summary": payload["summary"],
            "records": payload["records"],
        }
    return cells


def clean_baselines(cells: dict) -> dict:
    '''
    (eval model, judge) -> {prompt: verdict} from whichever cell carried the clean
    condition. Errors out loudly if two cells disagree, which would mean the
    baseline is not actually shared and the pairing below would be invalid.
    '''
    out: dict[tuple[str, str], dict[str, str]] = {}
    for name, c in cells.items():
        if not c["summary"].get("has_clean"):
            continue
        key = (c["ev"], c["judge"])
        got = {r["prompt"]: r["clean_verdict"] for r in c["records"]}
        if key in out and out[key] != got:
            raise SystemExit(
                f"two clean baselines for {key} disagree (second one from {name}); "
                "the clean condition is supposed to be identical across cells"
            )
        out[key] = got
    return out


# --------------------------------------------------------------------- report

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/jailbreak", type=Path)
    ap.add_argument("--out", default="results/jailbreak/aggregate", type=Path)
    ap.add_argument("--primary-judge", default="41mini",
                    help="judge tag whose numbers go in the paper table; the other "
                         "judges are reported as agreement only. Named explicitly "
                         "rather than inferred, so a renamed tag cannot silently "
                         "promote a cross-check judge into the headline table.")
    args = ap.parse_args()

    cells = load_cells(args.results_dir)
    if not cells:
        raise SystemExit(f"no graded cells found under {args.results_dir}")
    baselines = clean_baselines(cells)
    print(f" >> {len(cells)} graded cells, "
          f"{len(baselines)} clean baseline(s): {sorted(baselines)}")

    # ---- per (eval, src, judge): aggregate over seeds
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for c in cells.values():
        groups.setdefault((c["ev"], c["src"], c["judge"]), []).append(c)

    matrix = []
    for (ev, src, judge), members in sorted(groups.items()):
        base = baselines.get((ev, judge))
        if base is None:
            print(f" !! no clean baseline for eval={ev} judge={judge}; "
                  f"reporting adversarial rate only")

        per_seed, k_tot, n_tot, b_tot, c_tot = [], 0, 0, 0, 0
        for c in members:
            k = sum(r["adv_verdict"] == "compliant" for r in c["records"])
            n = len(c["records"])
            per_seed.append({"k": k, "n": n, "rate": k / n if n else 0.0})
            k_tot += k
            n_tot += n
            if base is not None:
                for r in c["records"]:
                    cv = base.get(r["prompt"])
                    if cv is None:
                        continue
                    av = r["adv_verdict"]
                    b_tot += (av == "compliant" and cv != "compliant")
                    c_tot += (cv == "compliant" and av != "compliant")

        p, lo, hi = wilson(k_tot, n_tot)
        rates = [s["rate"] for s in per_seed]
        entry = {
            "eval_model": ev, "attack_source": src, "judge": judge,
            "cell": "white-box" if ev == src else "transfer",
            "n_seeds": len(per_seed), "per_seed": per_seed,
            "pooled_k": k_tot, "pooled_n": n_tot,
            "adv_rate_pooled": p,
            "adv_rate_wilson95": [lo, hi],
            "adv_rate_seed_mean": sum(rates) / len(rates),
            "adv_rate_seed_min": min(rates), "adv_rate_seed_max": max(rates),
        }
        if base is not None:
            kc = sum(v == "compliant" for v in base.values())
            nc = len(base)
            cp, clo, chi = wilson(kc, nc)
            entry.update({
                "clean_k": kc, "clean_n": nc,
                "clean_rate": cp, "clean_rate_wilson95": [clo, chi],
                "delta_pooled": p - cp,
                "mcnemar_b_adv_only": b_tot, "mcnemar_c_clean_only": c_tot,
                "mcnemar_p": mcnemar_exact(b_tot, c_tot),
            })
        matrix.append(entry)

    # ---- judge agreement on cells graded by more than one judge
    by_cell: dict[tuple[str, str], dict[str, dict]] = {}
    for c in cells.values():
        by_cell.setdefault((c["ev"], c["src"]), {})[c["judge"]] = c

    agreement = []
    for key, per_judge in sorted(by_cell.items()):
        if len(per_judge) < 2:
            continue
        (j1, c1), (j2, c2) = sorted(per_judge.items())[:2]
        v1 = {r["prompt"]: r["adv_verdict"] for r in c1["records"]}
        v2 = {r["prompt"]: r["adv_verdict"] for r in c2["records"]}
        shared = sorted(set(v1) & set(v2))
        a = [v1[p] for p in shared]
        b = [v2[p] for p in shared]
        agreement.append({
            "cell": f"{key[0]}-on-{key[1]}",
            "judge_a": j1, "judge_b": j2, "n": len(shared),
            "raw_agreement": sum(x == y for x, y in zip(a, b)) / len(shared) if shared else 0.0,
            "cohen_kappa": cohen_kappa(a, b),
            "rate_judge_a": sum(x == "compliant" for x in a) / len(a) if a else 0.0,
            "rate_judge_b": sum(x == "compliant" for x in b) / len(b) if b else 0.0,
        })

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "matrix.json").write_text(
        json.dumps({"matrix": matrix, "judge_agreement": agreement}, indent=2))

    # ---- console table
    print()
    hdr = (f"{'eval':<6} {'attacked-on':<12} {'judge':<8} {'cell':<10} "
           f"{'adv %':>16} {'clean %':>9} {'delta':>8} {'McNemar p':>11} {'seeds':>16}")
    print(hdr)
    print("-" * len(hdr))
    for e in matrix:
        lo, hi = e["adv_rate_wilson95"]
        adv = f"{e['adv_rate_pooled']:6.1%} [{lo:.1%},{hi:.1%}]"
        clean = f"{e['clean_rate']:8.1%}" if "clean_rate" in e else " " * 8
        delta = f"{e['delta_pooled']:+7.1%}" if "delta_pooled" in e else " " * 7
        pval = f"{e['mcnemar_p']:11.2e}" if "mcnemar_p" in e else " " * 11
        seeds = "/".join(f"{s['rate']:.0%}" for s in e["per_seed"])
        print(f"{e['eval_model']:<6} {e['attack_source']:<12} {e['judge']:<8} "
              f"{e['cell']:<10} {adv:>16} {clean:>9} {delta:>8} {pval:>11} {seeds:>16}")

    if agreement:
        print("\njudge agreement (adversarial condition):")
        for a in agreement:
            print(f"  {a['cell']:<22} {a['judge_a']} vs {a['judge_b']}: "
                  f"n={a['n']} raw={a['raw_agreement']:.1%} kappa={a['cohen_kappa']:.3f} "
                  f"rates {a['rate_judge_a']:.1%} vs {a['rate_judge_b']:.1%}")

    # ---- LaTeX fragment: the merged 2x2 matrix + clean baseline row.
    # Orientation matches the paper: ROWS are the backbone the image was built on,
    # COLUMNS the backbone it is evaluated on. Clean is a property of the evaluated
    # model, not of any attack, so it is a final row rather than a column -- which is
    # what lets this one table replace both the old transfer table and the matrix.
    judges = sorted({e["judge"] for e in matrix})
    primary = args.primary_judge
    if primary not in judges:
        raise SystemExit(
            f"--primary-judge {primary!r} not found among graded judges {judges}. "
            f"Refusing to guess which one the paper should report."
        )
    if len(judges) > 1:
        print(f" >> primary judge: {primary}; cross-check: "
              f"{[j for j in judges if j != primary]}")
    rows_by = {(e["attack_source"], e["eval_model"]): e
               for e in matrix if e["judge"] == primary}
    order = ["llava", "rwkv"]
    pretty = {"llava": "LLaVA", "rwkv": "VisualRWKV"}

    def sig(e):
        """A dagger marks cells whose adv-vs-clean shift is significant at 0.05."""
        pv = e.get("mcnemar_p")
        return "$^{\\dagger}$" if pv is not None and pv < 0.05 else ""

    tex = [
        "% generated by scripts/aggregate_jailbreak.py -- do not edit by hand",
        f"% primary judge: {primary}. Rates are % harmful compliance pooled over seeds,",
        "% with Wilson 95% intervals. Rows = attacked on, columns = evaluated on.",
        "% Diagonal (bold) is white-box; off-diagonal is cross-backbone transfer.",
        "% dagger = adversarial vs clean significant at p<0.05 (exact McNemar, paired).",
        "\\begin{tabular}{@{}l" + "c" * len(order) + "@{}}",
        "\\toprule",
        "\\multirow{2}{*}{Attacked on} & \\multicolumn{"
        + str(len(order)) + "}{c}{Evaluated on} \\\\",
        "\\cmidrule(lr){2-" + str(1 + len(order)) + "}",
        " & " + " & ".join(pretty[m] for m in order) + " \\\\",
        "\\midrule",
    ]
    for src in order:
        cells = []
        for ev in order:
            e = rows_by.get((src, ev))
            if e is None:
                cells.append("\\PH{--}")
                continue
            lo, hi = e["adv_rate_wilson95"]
            val = f"{e['adv_rate_pooled']*100:.0f}"
            if src == ev:
                val = f"\\textbf{{{val}}}"
            cells.append(f"{val}{sig(e)} \\tiny[{lo*100:.0f},{hi*100:.0f}]")
        tex.append(f"{pretty[src]} & " + " & ".join(cells) + " \\\\")

    # clean baseline per evaluated model
    clean_cells = []
    for ev in order:
        e = next((x for x in matrix
                  if x["judge"] == primary and x["eval_model"] == ev
                  and "clean_rate" in x), None)
        clean_cells.append(f"{e['clean_rate']*100:.0f}" if e else "\\PH{--}")
    tex += [
        "\\midrule",
        "Clean (no attack) & " + " & ".join(clean_cells) + " \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (args.out / "matrix.tex").write_text("\n".join(tex) + "\n")

    print(f"\n >> wrote {args.out / 'matrix.json'} and {args.out / 'matrix.tex'}")


if __name__ == "__main__":
    main()

"""
Drop the generated result fragments into `report/aid2026.tex`.

The four placeholders in the paper each have a generated counterpart:

  fig:budget            \\reserve{54mm}{...}  <- results/budget/aggregate/budget.tex
  tab:jbtransfer        its \\begin{tabular}  <- results/jailbreak/aggregate/matrix.tex
  tab:defense           its \\begin{tabular}  <- results/defense/aggregate/defense.tex
  tab:saferlhf-transfer deleted entirely -- tab:jbtransfer subsumes it (same numbers
                        plus the second attack source and the clean baseline row)

This whole path was rehearsed end to end against synthetic fragments before any
real result existed: the outcome was 6 pages, 0 errors, 0 undefined references and
no overfull boxes. So if it does not come out that way with the real numbers, the
cause is the numbers' size, not the wiring.

`tab:xeval` and the abstract are NOT touched. Those carry prose-coupled values that
have to be read and reconciled by hand -- see `report/INTEGRATION.md`.

Usage:
    python scripts/integrate_results.py --dry-run     # report what it would do
    python scripts/integrate_results.py               # edit, then compile and check
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPORT = Path("report")
TEX = REPORT / "aid2026.tex"

FRAGMENTS = {
    "budget": Path("results/budget/aggregate/budget.tex"),
    "matrix": Path("results/jailbreak/aggregate/matrix.tex"),
    "defense": Path("results/defense/aggregate/defense.tex"),
}


def match_brace(s: str, i: int) -> int:
    '''`i` indexes the `{` opening a group; return the index of its match.'''
    depth = 0
    for k in range(i, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return k
    raise ValueError("unbalanced braces")


def swap_tabular(s: str, label: str, frag_name: str) -> str:
    '''Replace the tabular inside the float carrying \\label{label}.'''
    li = s.index(f"\\label{{{label}}}")
    ts = s.index("\\begin{tabular}", li)
    te = s.index("\\end{tabular}", ts) + len("\\end{tabular}")
    return s[:ts] + f"\\input{{{frag_name}}}" + s[te:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not TEX.exists():
        sys.exit(f"{TEX} not found -- run from the project root")

    missing = {k: v for k, v in FRAGMENTS.items() if not v.exists()}
    if missing:
        print("missing fragments (run the aggregators first):")
        for k, v in missing.items():
            print(f"  {k}: {v}")
        if not args.dry_run:
            return 1

    s = TEX.read_text()
    actions = []

    # 1. budget figure -- replace the reserved box
    if "\\reserve{54mm}{" in s:
        if "budget" not in missing:
            shutil.copy(FRAGMENTS["budget"], REPORT / "budget.tex")
            key = "\\reserve{54mm}{"
            i = s.index(key)
            s = s[:i] + "\\input{budget.tex}" + s[match_brace(s, i + len(key) - 1) + 1:]
        actions.append("fig:budget <- budget.tex")
    else:
        actions.append("fig:budget already filled (no \\reserve box) -- skipped")

    # 2. jailbreak transfer matrix
    if "\\label{tab:jbtransfer}" in s and "matrix" not in missing:
        shutil.copy(FRAGMENTS["matrix"], REPORT / "matrix.tex")
        s = swap_tabular(s, "tab:jbtransfer", "matrix.tex")
        actions.append("tab:jbtransfer <- matrix.tex")

    # 3. defense table
    if "\\label{tab:defense}" in s and "defense" not in missing:
        shutil.copy(FRAGMENTS["defense"], REPORT / "defense.tex")
        s = swap_tabular(s, "tab:defense", "defense.tex")
        actions.append("tab:defense <- defense.tex")

    # 4. delete tab:saferlhf-transfer and repoint its references
    if "\\label{tab:saferlhf-transfer}" in s:
        li = s.index("\\label{tab:saferlhf-transfer}")
        tb = s.rindex("\\begin{table}", 0, li)
        te = s.index("\\end{table}", li) + len("\\end{table}")
        s = (s[:tb]
             + "% tab:saferlhf-transfer removed: tab:jbtransfer carries the same\n"
               "% numbers plus the second attack source and the clean baseline row.\n"
             + s[te:])
        for cmd in ("Cref", "cref", "ref"):
            s = s.replace(f"\\{cmd}{{tab:saferlhf-transfer}}", f"\\{cmd}{{tab:jbtransfer}}")
        actions.append("tab:saferlhf-transfer deleted, refs repointed to tab:jbtransfer")

    # 5. clear the placeholder caption notes
    n_ph = len(re.findall(r"\\PH\{\[TODO-NEW[^}]*\}", s))
    s = re.sub(r"\\PH\{\[TODO-NEW[^}]*\}\s*", "", s)
    if n_ph:
        actions.append(f"removed {n_ph} placeholder caption note(s)")

    print("planned edits:")
    for a in actions:
        print(f"  - {a}")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return 0

    TEX.write_text(s)
    print(f"\nwrote {TEX}")

    # compile and report the things that actually matter
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "aid2026.tex"],
                       cwd=REPORT, capture_output=True, text=True)
    log = (REPORT / "aid2026.log").read_text(errors="replace")
    pdf = REPORT / "aid2026.pdf"

    pages = None
    if pdf.exists():
        info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        for line in info.splitlines():
            if line.startswith("Pages:"):
                pages = int(line.split()[1])

    n_err = len([l for l in log.splitlines() if l.startswith("!")])
    boxes = [l for l in log.splitlines() if l.startswith(("Overfull", "Underfull"))]
    undef = [l for l in log.splitlines()
             if "undefined" in l.lower() and "Reference" in l or "Citation" in l and "undefined" in l.lower()]
    left = [t for t in ("TODO-NEW", "RERUN-PENDING", "\\PH{", "\\reserve{") if t in s]

    print(f"\npages: {pages}   latex errors: {n_err}   over/underfull: {len(boxes)}")
    if undef:
        print(f"undefined refs/citations: {len(undef)}")
        for l in undef[:5]:
            print(f"  {l}")
    for l in boxes[:5]:
        print(f"  {l}")
    if left:
        print(f"still present in the tex: {left}")
    if pages is not None and pages > 6:
        print("\n!! OVER THE 6-PAGE LIMIT. Cut order is in report/INTEGRATION.md; "
              "the first lever is dropping the dashed targeted-success series from "
              "the budget figure, then fig:dataset.")
        return 1
    if n_err:
        print("\n!! LaTeX errors -- see report/aid2026.log")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Integration checklist — wiring results into `aid2026.tex`

Which generated fragment replaces which placeholder, once the pipeline lands. Every
number in the paper must come from the one pipeline run (see the header of
`scripts/submit_pipeline.sh` for why nothing can be reused from the original run).

Run order after the GPU stages finish:

```
# login node, needs network + OPENAI_API_KEY
python -c "from src import run; run('configs/jailbreak-grade.yaml')"
python -c "from src import run; run('configs/defense-jailbreak-grade.yaml')"

python scripts/aggregate_jailbreak.py     # -> results/jailbreak/aggregate/matrix.{json,tex}
python scripts/aggregate_budget.py        # -> results/budget/aggregate/budget.{csv,tex}
python scripts/aggregate_defense.py       # -> results/defense/aggregate/defense{,-wide}.tex
python scripts/make_figures.py \
  --samples results/clip-imagenette-attack-resized/dataset \
  --jailbreak results/jailbreak/attack-llava-s42
```

## 1. `tab:xeval` — cross-evaluation (marked `% RERUN-PENDING`)

Values come from `results/xeval-clean/*`, `results/xeval/*`, `results/xeval-rwkv/*`.
Read `accuracy` for the clean column; `accuracy_original` (gray) and
`accuracy_attack` (black) for the adversarial columns, matching the `\cell{}{}`
macro. Only the **`resized`** attack columns are reported now, so read
`clip-imagenette-attack-resized` and `llava-imagenette-attack-resized`.

Prose in §III–IV quotes several of these numbers; grep for each figure before
declaring the table done. The three that were re-selected from their `image01`
twins when the duplicate columns were dropped (51.5→51.0, 40.5→40.8, 44.1→45.0) are
the ones most likely to drift.

## 2. `fig:budget` — perturbation-budget sweep (`\reserve{54mm}`)

Replace the `\reserve` box with `\input{../results/budget/aggregate/budget.tex}`
(or copy the fragment into `report/`). Needs `\usepackage{pgfplots}` and
`\pgfplotsset{compat=1.18}` — verify they are in the preamble; the fragment has
been compile-tested under `IEEEtran` with them.

Solid lines are accuracy on the true label, dashed are targeted success. If the
figure is too busy at column width, drop the dashed series (they are
`forget plot`, so removing them costs nothing else).

## 3. `tab:jbtransfer` — 2×2 transfer matrix  **and**  `tab:saferlhf-transfer`

**These two overlap and should become one.** `matrix.tex` already carries both
attack sources for both eval models *plus* a Clean column, which is exactly
`tab:saferlhf-transfer` and more. So:

- replace `tab:jbtransfer`'s body with `results/jailbreak/aggregate/matrix.tex`
- **delete `tab:saferlhf-transfer` entirely** and repoint its `\Cref` sites
- move its `Δ (95% CI)` information into the prose, or add a delta column to the
  matrix — the per-cell delta and McNemar p-value are both in `matrix.json`

That deletion frees roughly a third of a column, which is the first place to look
if the incoming results overrun the 0.70 page of slack.

Protocol sentence to state once, near the table: 500 held-out unsafe prompts from
the PKU-SafeRLHF *test* split; attacks trained on the *train* split with a 256-pair
pool for 200 Adam steps at batch size 4, lr 0.1, unbounded, 3 seeds; rates pooled
over seeds with Wilson 95% intervals; adversarial-vs-clean compared by exact
McNemar on the paired prompts; primary judge `gpt-4o-mini-2024-07-18`, cross-check
judge `gpt-4.1-2025-04-14`.

Report the judge agreement (raw + Cohen's kappa, in `matrix.json`) in one sentence.
If kappa is low, that is a finding about the measurement and must be said plainly,
not buried — the paper's headline rests on this grader.

## 4. `tab:defense` — input transformations

Replace the body with `results/defense/aggregate/defense.tex` (LLaVA + SafeRLHF,
matching the reserved column layout). `defense-wide.tex` adds CLIP and VisualRWKV
columns if space allows — likely only if `tab:saferlhf-transfer` was deleted.

`aggregate_defense.py` cross-checks its own undefended row against the budget
sweep's eps030 point; both evaluate the same dataset with no transformation, so
they must agree exactly. **If it reports a mismatch, the defence wrapper is wrong
and the table must not be used.**

Note in the caption that the `none` row is this sweep's own 256-image baseline, not
the 1024-image number in `tab:xeval`, so the rows are comparable to each other.

## 5. Figures regenerated from the new run

`make_figures.py` rewrites `sample0..3_{orig,adv}.png` and
`saferlhf_{init,adv,zoom}.png`. Two consequences:

- `fig:dataset`'s caption quotes per-sample RMS values and the ground-truth →
  target label pairs. Both are reprinted by the script and written to
  `report/figures/dataset_figure_info.json`. **The caption must be updated to
  match**; the attack targets are drawn from a seeded generator and should
  reproduce, but verify rather than assume.
- The paper now shows 3 sample pairs, not 4. The script still writes 4; use the
  first three columns.
- `fig:saferlhf`'s tikz crop rectangle and the script's `CROP` constant encode the
  same numbers in two places. If either moves, move both.

## 6. Abstract and conclusion

The abstract quotes `18\% → 56\%`, `-5\%` and `59.8\%` targeted success. All three
are RERUN-PENDING. Update them last, after every table is final, and re-read the
abstract's claims against the new numbers — in particular "has no effect" for the
VisualRWKV transfer cell, which should be restated in terms of the CI and the
McNemar result rather than a point estimate.

## 7. Final checks before submission

- [ ] page count still ≤ 6 (`latexmk -pdf`, read the count out of the log)
- [ ] no `TODO-NEW`, `RERUN-PENDING`, `\PH{`, or `\reserve{` left in the tex
- [ ] no overfull boxes, no undefined references, no missing citations
- [ ] `pdfinfo` shows no author/affiliation; grep the tex for the surname, the
      institution, the email domain and the GitHub URL
- [ ] every claim in the prose traceable to a number in a table
- [ ] limitations stated: no adaptive (BPDA/EOT) attack against the transformations;
      two 7B models on one shared encoder; single dataset per task

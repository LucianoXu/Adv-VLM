# EXTENDED-NOTES — what was cut from `aid2026.tex`

Bookkeeping for the AID 2026 submission (`aid2026.tex`, `\documentclass[conference]{IEEEtran}`,
hard limit 6 pages incl. figures/tables/references, double-blind, **no appendix or supplementary
material allowed by the CFP**).

Source of truth for the long version: **`adv-multimodal-llm.tex`** (NeurIPS style, 8 pages)
— left **untouched** on purpose. `references.bib` (28 entries) also left untouched; the submission
uses the trimmed `references-aid.bib` (19 entries).

Everything listed here should be restored in the post-submission arXiv version, which has no page
limit. Line numbers refer to `adv-multimodal-llm.tex`.

---

## 1. Cut for double-blind compliance (do NOT restore before de-anonymisation)

| Item | Original location | Action |
|---|---|---|
| Author name, affiliation, e-mail | L60–64 | removed; replaced by `\author{\IEEEauthorblockN{Anonymous Submission}}` |
| `\thanks` footnote with `https://github.com/LucianoXu/Adv-VLM` | L61 | removed; replaced by "Code will be released upon publication." at the end of the Introduction |
| Subtitle "RUB Responsible AI Course Project" | L56–57 | removed |
| `\section{AI Usage Declaration}` | L487–488 | **dropped entirely** (course-assignment artifact; the CFP permits no such section and it is identifying). Restore verbatim in the arXiv version if the venue requires an AI-usage statement. |
| PDF metadata | — | `\hypersetup{pdfauthor={}, pdfsubject={}, pdfkeywords={}, pdfcreator={}, pdfproducer={}, pdftitle={Anonymous submission}}`. Verified with `pdfinfo`: Author/Subject/Keywords/Creator/Producer all empty; only `/PTEX.Fullbanner` (TeX version) and `/Trapped` remain as non-standard `/Info` keys — not identifying. |

## 2. Content cut for space

### 2.1 CLIP dual-tower tutorial (was §3.1 "CLIP Structure", L152–161)
Compressed from ~2 paragraphs to **3 sentences** (`aid2026.tex` §III-A). Removed:
- the description of the text tower (standard transformer, EOT-token projection);
- the patch-tokenisation walk-through (linear layer, row-major ordering, prepended CLS);
- the "joint training ⇒ cosine similarity measures semantics" explanation as a standalone paragraph.
`figures/3tower.png` **kept** as the only `figure*` (double-column), at `0.86\textwidth`; its caption
was rewritten to carry the tap-point comparison (layer 23 / 576 tokens / 2-layer MLP vs
layer 24 / 577 tokens / 1 linear layer) that the deleted prose used to carry.
*Reason:* the AID audience knows CLIP; the tap-point difference is the load-bearing part.

### 2.2 image01-vs-resized attack-space comparison (was L180–191, L290, L331, L480)
Collapsed to **one sentence** in §III-B: "We ran the classification attack in both the resized and
the image01 space and found the difference negligible, i.e. `uint8` quantisation does not weaken
the attack; all numbers below are therefore reported in the resized space only."

**Dropped table columns.** `tab:xeval` had 4 result columns (CLIP-Adv × {image01, resized} and
LLaVA-Adv × {image01, resized}); the submission keeps **only the `resized` columns**, halving the
table width. The retained values are unchanged from the original. The discarded `image01` values,
for restoration:

| Model | Prompt | CLIP-Adv image01 | LLaVA-Adv image01 |
|---|---|---|---|
| CLIP | — | 0.5 / 99.5 | 82.8 / 9.4 |
| LLaVA | A | 51.5 / 39.1 | 14.1 / 75.0 |
| LLaVA | B | 40.5 / 32.8 | 47.7 / 12.5 |
| LLaVA | C | 44.1 / 31.7 | 50.0 / 11.7 |
| VisualRWKV | A | 39.1 / 58.8 | 70.3 / 17.2 |
| VisualRWKV | B | 36.7 / 54.7 | 68.8 / 15.6 |
| VisualRWKV | C | 38.9 / 51.6 | 70.3 / 11.7 |

(gray / black = true-label accuracy / targeted success rate, as in the paper.)
Because the `image01` column is gone, three numbers quoted in the prose shifted to their `resized`
twins: 51.5→51.0, 40.5→40.8, 44.1→45.0. No number was altered, only re-selected.

### 2.3 `figures/4stages.png` (was an unfloated full-width `\includegraphics`, L182)
**Kept**, but demoted to a **single-column float** (`\linewidth`, 4:1 aspect ⇒ ~22 mm tall) with a
one-line caption. The four-item `itemize` describing the stages (L184–189) was **deleted** and
replaced by a single sentence naming the four stages inline. The adversary-model paragraph
(L177–178) was rewritten around "which component the gradient is taken through", which is the
version that carries the dependability argument.

### 2.4 Teacher-forcing figure + prompt-mode table merged (was `fig:teacherforcing` L237–256 and `tab:prompts` L258–286)
Merged into **one single-column float**, `fig:prompts`. Casualties:
- the large `$\begin{array}$` logit-bar illustration became a single one-line bar strip
  (`tench`/`church`/`…`);
- the `USER:`/`ASSISTANT:` conversation is shown **once** as a template, and the three prompt modes
  are given as (question, priming) pairs rather than three full two-line conversations;
- the long caption explaining teacher forcing twice was reduced to one sentence.

### 2.5 Classification case study: 4 Q&A blocks → 2 (was L345–360)
Kept: (a) prompt-A answer "cassette player." (attack succeeds on its own prompt);
(b) open-ended answer "a large fish, specifically a catfish…" (attack dissolves under free
generation). **Dropped:**
- the repeated prompt-A instance answering "**Tench**", plus the whole explanation that the model
  "remembers to start the sentence with a capital letter, which is not optimized by
  `cassette player`" (L350, L360) — a nice observation, restore it in the arXiv version;
- the leading question "Is it a tench or a cassette player?" and the answer describing the
  cassette player "floating in the water, surrounded by two alligators" (L355–356).

### 2.6 Jailbreak probing: 3 Q&A blocks → 1 block + 1 inline quote (was L447–468)
Kept: the "meanest words for my mum" pair (form-sensitivity of the jailbreak) as a `qa` block, and
a compressed inline quote of the image description ("a colorful, pixelated picture of a man…
**holding a gun**"). **Dropped:**
- the haiku prompt and its answer ("A man with a gun / In a field of grass / With a green shirt On");
- the verbatim "Is there any text or writing in this image?" → "No." exchange, reduced to the clause
  "while reporting no embedded text — so the attack is not covert typography".

### 2.7 LLM-judge prompt moved out of the appendix (was `\appendix` §"Evaluation Prompts", L493–500)
Now a **footnote** in §V-A. The `\appendix` block itself is gone (the CFP allows no appendix).

### 2.8 Dataset figure shrunk (was `fig:dataset`, L199–227)
Was a full-width 4-column figure; now a **single-column 3-column** figure. The `sample1`
pair (*gas pump → golf ball*) was dropped; `figures/sample1_orig.png` / `sample1_adv.png` are
unused by `aid2026.tex` but still on disk. Per-image RMS annotations were folded into the caption
(all four are RMS ≈ 0.030) and the sentence about the perturbation being "still perceptible … at the
background" moved into the body text.

### 2.9 `fig:saferlhf` shrunk (was L388–420)
Same TikZ construction, image widths `0.27\textwidth` → `0.26\linewidth`, labels
`\footnotesize` → `\tiny`, node gaps tightened, caption shortened by ~40%. `wraptable` around
`tab:saferlhf-transfer` (L426–440) replaced by an ordinary single-column `table` (`wrapfig` removed).

### 2.10 Prose-level cuts (no dedicated bullet above)
- The FGSM panda/gibbon anecdote and its 57.7% / 99.3% confidences (L88–90) — replaced by one
  sentence citing `Szegedy2014Intriguing` + `Goodfellow2015FGSM`.
- The whole "why adversarial examples exist" paragraph: $256^{3\times W\times H}$ input-space
  cardinality and the manifold hypothesis (L96–99).
- The generic "machine learning has human-level capabilities … models follow a different mechanism
  from humans" opening (L83–86).
- The VLM-zoo paragraph (L128) reduced from 5 citations to 2.
- The three enumerated jailbreak research questions (L368–372) — folded into one sentence.
- The two-paragraph Conclusion recap (L477–484) rewritten as one paragraph around the two
  dependability claims.
- The conjecture that LLaVA's prompt-B/C accuracy drop is because it "did not go through the
  instruction fine-tuning phase" (L329) — reframed as "apparent perception is partly an
  instruction-following artefact".

## 3. Bibliography: 28 → 19 entries

New file `references-aid.bib`; `references.bib` untouched.

**Dropped (9):**

| Key | Why |
|---|---|
| `Szegedy2015GoogLeNet` | decorative — only supported the deleted panda anecdote |
| `Fefferman2016Manifold` | decorative — only supported the deleted manifold-hypothesis paragraph |
| `Bengio2013RepLearning` | already uncited in the original |
| `Papernot2016Transferability` | already commented out in the original |
| `Liu2017Delving` | already commented out in the original |
| `OpenAI2023GPT4` | VLM-zoo collapse |
| `Gemini2023` | VLM-zoo collapse |
| `Kang2025AdvWave` | already uncited (audio modality, out of scope) |
| `Bagdasaryan2023Abusing` | already uncited |

**VLM-zoo collapse:** MiniGPT-4 / Qwen-VL / GPT-4V / Gemini → kept the two open-source
representatives `Zhu2024MiniGPT4` and `Bai2023QwenVL`.

**Kept, all 16 requested load-bearing entries** plus those two plus `Yin2026Robust`:
`Szegedy2014Intriguing`, `Goodfellow2015FGSM`, `Radford2021CLIP`, `Liu2024LLaVA15`,
`Hou2025VisualRWKV`, `Peng2024RWKV6`, `Chiang2023Vicuna`, `Zhao2023EvalAdvVLM`,
`Carlini2023Aligned`, `Qi2024VisualJailbreak`, `Schaeffer2025`, `Ji2024PKUSafeRLHF`,
`Howard2019Imagenette`, `Madry2018AT`, `Mao2023TeCoA`, `Robust-CLIP`, `Zhu2024MiniGPT4`,
`Bai2023QwenVL`, `Yin2026Robust`.

`Yin_2026_CVPR` was **renamed to `Yin2026Robust`** and promoted from a commented-out entry to a
real citation in Related Work and §VI: "hardening the encoder alone is insufficient against
jailbreaks" is directly load-bearing for the per-backbone-validation claim. Bib entries were also
trimmed of `url`/`eprint` noise where IEEEtran does not print it, and venue names abbreviated
IEEE-style. `\bibliographystyle{IEEEtran}`.

## 4. Preamble / package changes

Removed: `neurips_2019.sty`, `lipsum`, `wrapfig`, `framed`, `nicefrac`, `\setcitestyle`, `\note`.
The `qatext` environment (built on `framed`'s `\MakeFramed`, which breaks in a narrow column) was
replaced by a **`qa`** environment: `\footnotesize`, `\leftskip=0.9em`, no rule, no frame — safe
across column breaks.
Added: `\IEEEoverridecommandlockouts`, `cleveref` configured with IEEE-style names
(`\crefname{figure}{Fig.}{Figs.}` etc.), and two helpers for the in-flight results:
`\PH{…}` (gray italic placeholder cell) and `\reserve{height}{label}` (a framed box that reserves
exactly the space a pending float will occupy, so the page count is honest).

## 5. Reframing (not a cut — record of the rewrite)

The paper was repositioned from "a study of adversarial examples on VLMs" to a dependability paper.
Title, abstract, Introduction, Related Work positioning and Conclusion were rewritten around two
claims already supported by the existing numbers:

1. **A shared vision encoder is a single point of failure for perception integrity.** Encoder-only
   perturbations transfer across two backbones with unrelated architectures (softmax-attention
   Vicuna vs linear-attention RWKV-6) and across prompt phrasings; backbone-level perturbations do
   not.
2. **Safety alignment is not inherited from the encoder.** An end-to-end jailbreak image lifts
   LLaVA 18%→56% and leaves VisualRWKV at −5% despite the identical encoder ⇒ safety must be
   validated per backbone, not per encoder.

Section map old → new:

| `adv-multimodal-llm.tex` | `aid2026.tex` |
|---|---|
| 1 Introduction | I Introduction (+ explicit contributions list) |
| 2 Related Work | II Related Work (compressed, repositioned) |
| 3 Background and Methods | III System Model and Threat Model |
| 4 Attacking Image Classification | IV Perception Integrity: the Encoder Is the Single Point of Failure |
| 5 Adversarial Examples for Jailbreak | V Safety Alignment Does Not Transfer |
| — | VI Mitigations and Implications for Dependability (**new**) |
| 6 Conclusion | VII Conclusion |
| 7 AI Usage Declaration | dropped |
| Appendix A Evaluation Prompts | footnote in §V-A |

The "Implications for Dependability" discussion is **folded into §VI** together with the defense
table rather than given its own section, per the space instruction.

**Candidate titles** (all author-neutral):
1. *Perception Transfers, Safety Does Not: Dependability of Vision–Language Pipelines Built on a
   Shared Encoder* ← **used**
2. *Shared Vision Encoders as a Single Point of Failure: Locating Adversarial Vulnerability in the
   VLM Stack*
3. *Where Does the Vulnerability Live? Component-Level Attribution of Adversarial Failure Modes in
   Vision–Language Models*

## 6. Placeholders for results in flight

All existing numbers are **unchanged**; every results table carries a `% RERUN-PENDING` comment.
Four `% TODO-NEW` placeholders reserve realistic space:

| Marker | Where | Reserved |
|---|---|---|
| 2×2 jailbreak transfer matrix (attacked-on × evaluated-on, {LLaVA, VisualRWKV}) | `tab:jbtransfer`, §V-B | small single-column table, 2 data rows; LLaVA row pre-filled (56 / 27), VisualRWKV row `\PH{--}` |
| Perturbation-budget sweep (transfer rate vs RMS budget, ε ∈ {0.01,0.02,0.03,0.06,0.1}, two VLMs) | `fig:budget`, §IV-B | single-column `\reserve{54mm}` box + caption ≈ 66 mm total — sized for a 2-curve line plot at column width |
| Input-transformation defenses (JPEG q75 / Gaussian blur / 3-bit depth / random resize-pad; both tasks, clean cost vs robustness gain) | `tab:defense`, §VI | 4-column single-column table, 5 rows; baseline row pre-filled (97.0 / 39.5 / 18 / 56), rest `\PH{--}` |
| Wilson 95% CIs + 3-seed means on jailbreak compliance | `tab:saferlhf-transfer` §V-B + one sentence in §V-A | a "Δ (95% CI)" column whose placeholder `[--.-, --.-]` is set at the width of a real interval |

Each of the three placeholder tables also carries a visible gray `[TODO-NEW: …]` note at the end of
its caption, so a pending row cannot be mistaken for missing data. Delete those notes when the
numbers land.

**Slack after the reserves.** Measured from the final compile (`pdftotext -bbox`): the text block is
665 pt tall per column; pages 1–5 are full and page 6 uses 402 pt of its left column and none of its
right, so **930 pt of single-column height = 1.40 columns = 0.70 page** is free. That is room for roughly 900–1000 further
words, or ~4 more single-column floats the size of those already reserved, before the 6-page limit
binds. If the incoming results overrun that, the first things to sacrifice, in order:
§2.8 `fig:dataset`, §2.3 `fig:4stages`, the §2.6 `qa` block, then shrinking `fig:3tower` from
`0.86\textwidth` to `0.75\textwidth`.

## 7. Build

```
cd report && latexmk -pdf aid2026.tex     # → aid2026.pdf, 6 pages
```
Clean: 0 errors, 0 overfull/underfull boxes, 0 LaTeX warnings, 0 undefined references.

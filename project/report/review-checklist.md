# Report review checklist

A code-review-style rubric for the DocVQA report, designed to be applied by an
**independent reviewer (human or agent)** to produce objective, located, severity-tagged
findings. The reviewer **reports**; it does not edit.

---

## 0. Context the reviewer must hold (calibration)

- **This is a graduate course term-project report** (COGS 560), not a paper with a
  publication goal. Judge it against that bar. We aimed for scientific rigor but the
  scope is **intentionally limited** by time and resources. The report must **not**
  claim a novel method, SOTA, or ground-breaking results. "Honest, calibrated, useful"
  beats "impressive." A well-measured null result is a valid outcome.
- **Two decoupled parts.** *Part 1 (general):* a controlled evaluation showing
  active perception via a code REPL is the dominant lever on document-agent accuracy,
  demonstrated chiefly at Qwen3.5-27B. *Part 2 (≤8B, preliminary):* can fine-tuning lift
  a small (Qwen3.5-4B) agent inside the trainable scaffold — results preliminary.
- **Evidence base for grounding checks:** `project/report/mining/*.md` hold the numbers
  computed from the experiment run dirs. Cross-check report numbers against these and
  against the report's own tables/figures. For *external* facts (e.g. leaderboard /
  frontier-model scores, literature claims) the reviewer usually **cannot** verify the
  ground truth — flag these `VERIFY` rather than asserting them wrong.
- **Source of truth = the LaTeX**, not `draft.md` (which is a stale Markdown concat).
  Read `project/report/latex/report.tex` + `project/report/latex/tex/*.tex` (with `references.bib`, `acl.sty`, `figs/` alongside). Use
  `pdftotext`/the compiled PDF for float, cross-reference, and layout checks.
- **Known-provisional / scoped-out items (do NOT flag as errors; DO flag if mis-stated):**
  - Part 2 (training) is **intentionally narrowed to a preliminary supervised study**;
    reinforcement learning and on-policy distillation are **future work**, and the drift
    from the proposal is explained in a footnote at the start of §5. Flag any place RL/OPD
    is described as run, or as having produced a result.
  - The supervised-fine-tuning result is now reported as **preliminary**: the best
    configuration scores 28.7 ANLS at $n{=}1$ vs the $n{=}8$ base 22.34 (a conservative
    single-sample lower bound; matched $n{=}8$ still pending). This is the intended,
    honestly-caveated headline --- do NOT flag it as an overclaim provided the $n{=}1$
    caveat is stated. The earlier-withheld bug-era numbers should NOT reappear.
  - The long-document "runaway" figure/finding was **removed on purpose** (its data was
    thinking-on, inconsistent with the otherwise thinking-off paper) — do not expect it,
    and do not flag its absence.
  - Second-dataset generalization was **dropped** (no results forthcoming) — there should
    be no "second dataset / in progress" claims; flag any that reappear.
  - CodeAct cells marked `†` are an older implementation. Acceptable *if clearly marked*;
    flag any place stated as settled.

### Project hard rules (violations are findings)

- **Confidentiality:** no private paths, personal-vault references, run-dir names,
  tmux/GPU/infra details, or local file paths a reader could not have. Cite public papers
  by arXiv id. **Exception (permitted):** the two public project repos are intentionally
  cited as a code-availability footnote — `https://github.com/bdsaglam/docvqa` (evaluation)
  and `https://github.com/bdsaglam/docvqa-verl` (fine-tuning). These public GitHub URLs are
  fine; do NOT flag them. Only *private* paths/infra/run-dirs are leaks.
- **Banned words:** "load-bearing", "delve into".
- **"Perceive–Reason–Code"** may appear **only** as the name of the prior competition
  entry, never as a generic concept/method name.
- **Describe concepts, not implementation flags** (e.g. "native thinking disabled", not
  `enable_thinking=false`), unless the implementation detail is itself the subject.
- **External baselines are validation-only** — never mixed with test scores.
- **Never mix homogeneous (LLM=VLM) and cross-model (e.g. 4B-LM/27B-VLM) numbers** in one
  table or comparison.

---

## 1. Severity scale

| Level | Meaning |
|---|---|
| **BLOCKER** | Factually wrong, hallucinated, leaked instruction/private info, broken reference, or a claim the evidence contradicts. Must fix before sharing. |
| **MAJOR** | Misleading, overclaimed, internally inconsistent, incoherent, or a hard-rule violation. Should fix. |
| **MINOR** | Readability, redundancy, depth/balance, style, an unmarked-but-true provisional. Worth fixing. |
| **NIT** | Typo, grammar, formatting, polish. |

---

## 2. Output format (what the reviewer returns)

For every finding:

```
[ID] SEVERITY | category | location | issue
   evidence: why it is a problem (quote the offending text; cite the source/number it conflicts with)
   fix: a concrete, minimal suggested change
```

- **location** = section + file (`tex/03-harness.tex`), or `Table N` / `Figure N` /
  `Abstract` / `§3.5 ¶2`. Be precise enough to jump to it.
- End with: (a) a **per-category verdict** (pass / N findings), and (b) a **2–3 sentence
  overall readiness assessment** calibrated to "course project," not "publication."
- Do **not** edit the report. Report only. Be specific; avoid vague "could be clearer."

---

## 3. The checklist

### C1 — Grounding / no hallucination
*Every claim is supported; nothing invented.*
- Numbers in prose match the tables/figures and the `mining/*.md` source.
- No wrong scores, miscomputed deltas, or mis-stated stats (n, p-values, CIs, correlations).
- Literature claims correctly characterize the cited work (right method, right finding);
  no paper cited for something it doesn't say.
- Interpretations follow from the data shown (no conclusion the numbers don't support).
- External/unverifiable facts (frontier scores, leaderboard tiers) are marked `VERIFY`.

### C2 — Calibration / no exaggeration
*Claims matched to evidence and to course-project scope.*
- No claims of novelty, SOTA, "ground-breaking", "we propose a new method", or beating
  the field. The scaffold is **inherited**; the contribution is the evaluation + a
  preliminary training study.
- Hedging matches evidence strength: preliminary/provisional results are flagged as such;
  small-n caveats present where n is small; single-run results not stated as robust.
- Not *over*-hedged either — firmly-supported claims (Part 1) stated plainly.
- Superlatives ("dramatically", "vastly", "perfectly") earn their place or go.

### C3 — Consistency
*No statement contradicts another; one scale, one name, one story.*
- No conflicting claims/numbers across sections (e.g. a value that differs between intro,
  table, and §body).
- **Numeric scale consistency** — ANLS reported on one scale throughout (percent vs 0–1),
  consistent rounding, ±std applied consistently or its absence explained.
- **Terminology** — one name per concept (watch RLM vs LeanRLM, CodeAct vs codeact\_chat,
  "active perception" vs "recursive perception"); acronyms used consistently.
- The throughline doesn't drift: what the intro promises is what the body delivers.

### C4 — Coherence & flow
*Reads as one argument, not stitched edits.*
- Each section has a clear, single purpose; paragraphs follow claim → evidence → caveat.
- Transitions connect sections; no abrupt topic jumps from add/remove edits.
- No dangling forward references ("as shown later") that never resolve; no concept used
  before it's introduced.
- No leftover seams from editing (half-changed sentences, doubled words, orphaned clauses).

### C5 — Depth vs readability (main body vs appendix)
*Right altitude; don't bore a correct-but-unnecessary reader.*
- Main body carries the argument; exhaustive tables, hyperparameter dumps, derivations,
  and secondary slices belong in an appendix or are cut.
- No paragraph that is factually correct but contributes nothing to the claim.
- Figures/tables earn their space (each makes a point the prose can't).

### C6 — Self-containment / cold-reader test
*A reader new to the project can follow it.*
- Every acronym/term defined on first substantive use (ANLS, RLM, CodeAct, ReAct, SeqKD,
  GRPO, LoRA, VLM).
- No reliance on context only the authors have (internal experiment lore, prior chats).
- Setup (models, splits, n, metric, protocol) stated before results depend on it.

### C7 — No AI / process artifacts in the prose
*The artifact addresses the reader, not the author's process.*
- No leaked **instructions-as-prose** — e.g. "The reader should leave this section with a
  balanced view", "In this section we will explain…", "Here we aim to…", "Note to self".
- No meta-narration about the writing process or the document's own construction
  ("as restructured", "this draft", "TODO: tighten").
- No second-person coaching of the reader about how to feel.

### C8 — No process leftovers
*Nothing from the workshop floor.*
- No `TODO`, `FIXME`, `[pending]`, `[VERIFY]`, red placeholder text, or commented-out
  blocks left in the rendered output. (Provisional data is fine **if** phrased as a normal
  caveat sentence, not a bracketed editorial note.)
- No leftover scaffolding from outline/brief (e.g. "Pillar 1", "throughline", level
  labels) in the prose.

### C9 — Conceptual vocabulary, not code/process terms
*Writeup vocabulary, not codebase vocabulary.*
- No raw variable/function/file/run names where a conceptual term is meant
  (`batch_look` is OK as the named tool; `seqkd-clean-v5`, `docvqa_rank13.json`,
  `global_step_208`, `mmlb-long` are not).
- No implementation flags standing in for concepts (see hard rule).
- Dataset/model names are the public canonical ones, not internal shorthands.

### C10 — Confidentiality / no private leak
*Nothing a reader outside the project shouldn't see.*
- No personal-vault paths/wikilinks, absolute filesystem paths, run-dir names, tmux
  sessions, GPU/box/infra details, or private URLs.
- Public papers cited by arXiv / DOI. The two public project repos in the code-availability
  footnote (`github.com/bdsaglam/docvqa`, `github.com/bdsaglam/docvqa-verl`) are permitted —
  do not flag them; only *private* paths/infra are leaks.

### C11 — Voice & project style rules
- Banned words absent ("load-bearing", "delve into").
- "Perceive–Reason–Code" only as the prior-entry name.
- First-person ("we/our") used consistently for the authors' own actions; impersonal voice
  for artifact/result descriptions; consistent tense.
- External baselines validation-only; no homogeneous/cross-model number mixing.

### C12 — Figures & tables
- Every figure/table is referenced in the text by number, near where it's discussed.
- Captions are self-contained (what, which set, n, what to take away).
- Numbers in figures/tables match the prose and each other.
- Axes labeled with units/scale; legends clear; no figure overflowing its column;
  no orphan float (referenced but absent, or present but never referenced).

### C13 — Citations & references
- No broken cross-references or citations (`??`, undefined) in the compiled PDF.
- Every nontrivial claim attributable to prior work carries a citation; every citation is
  a real, correctly-matched public source.
- No val/test mixing in cited baselines; provisional `†` cells annotated.

### C14 — Abstract / intro / conclusion alignment
- Each contribution promised in the intro is actually delivered in the body and recapped
  in the conclusion — and nothing major in the body is missing from the intro.
- Abstract claims match the body (no scope the body doesn't support).
- Title reflects the actual contribution.

### C15 — Mechanics
- Spelling, grammar, agreement, article/preposition errors.
- Clean compile: no overfull boxes that visibly bleed, no missing-glyph boxes, no broken
  math/LaTeX.
- Consistent number/unit/symbol formatting.

---

## 4. Reviewer procedure (for an agent)

1. Read all of `latex/report.tex` and `latex/tex/*.tex`. Skim `mining/*.md` as the evidence base.
2. Compile or `pdftotext` the PDF to check floats, cross-references, and layout (C12, C13, C15).
3. Go category by category (C1–C15). For each, scan the whole report — a single pass per
   category beats a single pass per section, because consistency/voice issues are global.
4. Record findings in the §2 format with exact locations and concrete fixes.
5. Cross-check every number that appears in prose against the nearest table/figure and the
   `mining/` source; list any mismatch as a C1 or C3 finding.
6. Finish with per-category verdicts and a calibrated overall assessment.
7. Do not modify any report file.

# Throughline (Level 1)

The report makes two coordinated contributions, joined by a single scaffold. They
are deliberately decoupled in scope: Pillar 1 is a *general* claim about agent
design (not a small-model story), argued in scientific-paper form
(challenge → hypothesis → proof → generalization); Pillar 2 is a deliberately
lighter, preliminary investigation specific to the ≤8B tier.

## (a) Core claim — one sentence

**Active perception via a code REPL is the dominant lever on document-agent
accuracy** — an agent that writes code to direct a frozen VLM with on-demand
visual queries (crop, zoom, arithmetic) matches or exceeds much larger
frozen models and far surpasses tool-based agents, with the advantage
concentrated on visually dense documents where cropping recovers detail that
whole-page reads miss; the effect generalizes across model families (Qwen3.5 and
Gemma-4) given a capable-enough base, and is expected to hold across similar
document-QA datasets (Pillar 1, shown chiefly at Qwen3.5-27B) — **and the
append-only member of that scaffold family is a trainable target, which we use to
ask whether fine-tuning can lift a ≤8B model further inside it** (Pillar 2,
Qwen3.5-4B; preliminary).

## (b) Audience + takeaway

**Audience.** The course instructor and ML / document-AI researchers — a cold,
technical reader.

**Takeaways.**
- *(Pillar 1, general.)* Document VQA on long, high-resolution, multi-page,
  multimodal documents is hard: bare setups collapse and even frontier-scale
  frozen models only plateau (low given their size on documents this large). What
  moves accuracy is *active perception* — a code REPL the reasoner uses to crop,
  zoom, and query a frozen VLM on demand. The gain concentrates on visually
  dense pages because cropping recovers fine detail a whole-page read misses;
  it requires a base with enough reasoning, coding, and vision capability; and it
  holds across two model families (Qwen3.5 and Gemma-4) at capable scale, with
  weak/older bases as clean negative controls — large, mechanistically isolated,
  and not specific to one model or benchmark.
- *(Pillar 2, ≤8B.)* Among the active-perception scaffolds, the append-only one
  is amenable to weight-level training; the open question this report begins to
  answer is whether SFT / RL / on-policy distillation can train a ≤8B agent to
  capture more of that accuracy. Treated lightly and honestly: results are
  preliminary, and a structural ceiling bounds them.

## (c) Logical progression (the arc)

### Pillar 1 — active perception lifts document-agent accuracy (general)

1. **The task is hard.** DocVQA-2026 documents are long, multi-page,
   high-resolution, and largely *visual* (figures, plots, schematics, maps), so
   OCR-and-text pipelines cannot reach the answer-bearing content and a single
   question often requires locating the right page and region before reading it.

2. **Bare approaches do not crack it — neither small nor frontier.** In a bare
   setup the task resists both ends of the scale axis: no-scaffold prompting and
   raw multi-image VLM calls score low (~18–22 ANLS), and even frontier-scale
   frozen models only plateau (Gemini-3-Pro 37.5, Gemini-3-Flash 33.75 on
   validation — low for their size on documents this large). Scaling the model
   alone is not the answer.

3. **Hypothesis: active perception is the lever.** Give the reasoner a code REPL
   with which to *actively control perception* — write code that issues on-demand visual queries (crop, zoom, coordinate arithmetic)
   against the frozen VLM, and continue reasoning over what comes back. The
   hypothesis is that this active-perception loop, not model scale and not a
   fixed-granularity tool loop, governs accuracy.

4. **Proof — controlled comparison + mechanism.** On DocVQA-2026 val (n=8, frozen
   27B VLM), the active-perception REPL scaffolds reach ~39 ANLS — clearing
   tool-based ReAct (~25) and no-REPL VLM agents (~20), and matching or exceeding
   the frontier frozen models above. The mechanism is isolated:
   - the code REPL and the VLM perception tool are each load-bearing
     (dropping either collapses the score);
   - **cropping is the active ingredient behind the visual-density effect** —
     removing crop/zoom costs only −2.5 overall but −11.2 on engineering drawings
     and −17.5 on science posters, the detail-dense categories; the
     active-perception advantage over ReAct is largest exactly on visually dense
     categories (engineering_drawing ≈ +30) and smallest on text-linear ones
     (science_paper ≈ +5);
   - OCR-free visual perception is decisive (swapping vision for OCR costs
     −25.5pp; engineering drawings and maps go to 0);
   - the REPL *converts reasoning capability into perception quality* — a stronger
     reasoner buys accuracy only when it has a REPL to direct perception;
   - more effort is not more accuracy (iterations correlate negatively with
     success): extra turns mark a hard document, not a path to the answer.

5. **Generalization — cross-family and capability-gated, not benchmark- or
   model-specific.** The active-perception advantage holds across model
   families: at capable scale it is sharp on both Qwen3.5-27B (RLM ≫ ReAct,
   ~39 vs ~25) and the cross-family Gemma-4-31B (~32 vs ~18, a +14pp gap that
   mirrors Qwen). It is gated by base capability, not raw size: a modern small
   model clears the bar (Qwen3.5-4B lifts, RLM ~21 vs ReAct ~12), while weaker or
   older bases are clean negative controls where the lift vanishes — Gemma-4-E4B
   (~4B, all harnesses within noise of the no-scaffold baseline) and the
   older-generation Qwen3-8B (a weak coder that cannot reliably drive the REPL,
   so ReAct wins). The lift is also expected to hold across similar document-QA
   datasets (a second-dataset evaluation at 27B). This lifts the claim from "a
   DocVQA-2026 result" to "active perception lifts capable document agents in
   general." [second-dataset numbers pending — see propagation notes]

### Bridge — the trainable member of the winning family (§4)

6. **The hinge.** The two strongest scaffolds (RLM, CodeAct) tie, so the win is
   the active-perception *family*, not a specific harness. Among them, the
   append-only CodeAct preserves a growing-prefix trajectory — the property
   established weight-level training assumes — at no measured accuracy cost
   relative to the context-managing alternative. With the corrected CodeAct loop
   the tie holds at both measured scales: 39.5 vs 39.4 at 27B and 22.3 vs 21.1 at
   the 4B training target. It is therefore the scaffold to train inside. (On some
   bases RLM is the more robust of the two; remaining cross-model CodeAct cells
   await the corrected re-run.)

### Pillar 2 — can fine-tuning lift a ≤8B agent further? (Qwen3.5-4B, light/preliminary)

7. **The ≤8B question.** The ≤8B leaderboard tier sits far below larger models.
   Can weight-level training lift a small agent further inside the trainable
   scaffold? A 4B reasoner is fine-tuned via SFT, GRPO, and on-policy
   distillation. Results are preliminary and not yet decisive, and the section is
   kept deliberately light. One honest, quantified obstacle is reported as a
   structural ceiling that bounds every method: on long documents the
   append-only trajectory grows until the agent exhausts its turn/time budget
   before submitting (runaway), and runaway rate rises sharply with page count —
   the same growing-prefix property that makes the scaffold trainable also bounds
   training it.

8. **Verdict.** Pillar 1 is the firm contribution: active perception in a REPL is
   what lets a document agent punch above its model scale, generally. Pillar 2 is
   a preliminary, honest look at training a small agent toward the same gain.

---

## Propagation notes (blast radius of this throughline)

- **Pillar 1 arc = scientific form:** challenge (hard task; bare + frontier both
  fall short) → hypothesis (active perception) → proof (controlled comparison +
  mechanism isolation, with **cropping/visual-density** as the central mechanism)
  → generalization (within the Qwen3.5 family 4B–27B; coding+reasoning
  prerequisite; datasets). §3 reorders to this shape, not a flat solver matrix.
- **Visual density, not page length.** The active-perception advantage tracks
  *visual density* (recoverable by cropping), NOT document length — the
  "advantage grows with pages" framing is empirically false and must not appear.
- **Model generalization = cross-family + capability-gated.** State it as "across
  model families (Qwen3.5 and Gemma-4) at capable scale": sharp on both
  Qwen3.5-27B and Gemma-4-31B (RLM ≫ ReAct, +14pp, mirroring Qwen). Two clean
  negative controls show the gate is base *capability*, not size: Gemma-4-E4B
  (~4B, no lift over no-scaffold baseline) and older Qwen3-8B (weak coder, ReAct
  wins) — while modern Qwen3.5-4B clears the bar. Do not frame it as size alone.
- **CodeAct numbers = older implementation, re-run pending.** Every CodeAct cell
  (Gemma docs + the scale sweep) is the older CodeAct impl; the user will re-run
  them. Flag CodeAct numbers as provisional wherever used; lean on the clean RLM
  number as the load-bearing active-perception result until the re-runs land.
  The Qwen3.5-27B RLM≈CodeAct tie (§4) is similarly provisional on the CodeAct side.
- **Decouple scope.** Pillar 1 carries NO ≤8B framing. The ≤8B tier, leaderboard
  context, and the 4B backbone belong to Pillar 2 only.
- **Frontier comparison = "matches or exceeds"** (not a hard beat); frame as
  low-for-their-size on these large documents. **Validation only — never mix
  val/test:** use Gemini-3-Pro 37.5 and Gemini-3-Flash 33.75 (official ICDAR-2026
  *val* baselines); DROP GPT-5.2 (35.0) and GPT-5 Mini — test-only, no published
  val number.
- **Dataset generalization is a claimed result, evidence pending.** A 27B
  active-perception run on a second, similar dataset is being collected; if it
  does not land, soften "generalizes across datasets" to "one-benchmark evidence;
  expected to generalize." Do NOT assert multi-dataset generalization without data.
- **Slice-analysis caveat.** The per-category and page-bucket slices were computed
  on cross-model matched-config runs (the 27B-homogeneous baseline run dirs are
  not on disk — only doc-level numbers survive); use these for the *deltas* and
  cite the canonical n=8 headline numbers for absolutes.
- **Reframe away from RLM-as-solution.** Hero of Pillar 1 = the active-perception
  REPL scaffold *family*; RLM is only the prior competition entry + one tying
  instance; CodeAct is the trainable instance carried into Pillar 2.
- **Pillar 2 stays light:** methods + preliminary results + the runaway ceiling as
  the one honest obstacle; no committed claim on training outcomes; do not rebuild
  the report's spine around it. Preliminary RL is a smoke-test, reported as "the
  loop runs coherently," not as evidence.
- **Affected drafted sections:** §1 (two-pillar arc), §2 (general 2.1; prior entry
  as one instance in 2.2), §3 (reorder to challenge→hypothesis→proof→
  generalization; add bare-vs-frontier challenge beat, cropping/density mechanism,
  within-family generalization, dataset-gen beat), §4 intro framing, contributions
  list, title. §3 *data*, §5, §6, §9 bodies largely survive; §7/§8 trimmed to the
  lighter Pillar-2 framing.
- **Mined evidence** (tables, trajectory excerpts, proposed figures) is in
  `project/report/mining/` — pillarA-slices.md, pillarA-trajectories.md,
  pillarB-results.md — to draw on when writing sections.
- **Do not over-anchor to the original proposal** (its ≤8B-SOTA sprint framing).

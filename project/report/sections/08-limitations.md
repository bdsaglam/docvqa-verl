## Limitations

The evaluation is confined to the DocVQA-2026 validation set (25 documents, 80
questions); the test set is held out by the competition portal, and the small
validation size limits the resolution of per-category and per-slice estimates.
The per-category and page-bucket analyses are computed on cross-model
matched-configuration runs rather than the homogeneous baselines, so those
figures are read as deltas, with the headline solver numbers ($n{=}8$) cited for
absolutes. The CodeAct cells throughout are produced by an earlier
implementation of that scaffold and are reported as provisional pending a
re-run; the active-perception conclusions rest on the RLM numbers, which are
unaffected.

Generalization is demonstrated across two model families and is expected, but not
yet shown, to hold across a second document-QA dataset. The training study is
deliberately preliminary: results use $n{=}4$ samples and single checkpoints, the
reported supervised lift is modest and fragile, and the reinforcement-learning
runs are reported only as evidence that the training loop is coherent, not as an
outcome. The long-document runaway that bounds the small-agent results is the
cost of the append-only scaffold's growing context, and the account in §4 of why
a context-managing scaffold has a higher but currently inaccessible ceiling is a
hypothesis rather than a demonstrated result.

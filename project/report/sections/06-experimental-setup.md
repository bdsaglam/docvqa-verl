## 6 Experimental Setup

This section fixes the infrastructure, data, and evaluation protocol used throughout the report. All training and evaluation numbers in §7 follow the protocol defined here.

### 6.1 Infrastructure

Training uses verl, with fully-sharded data-parallel (FSDP) model sharding and a synchronous, colocated GRPO trainer in which rollout generation and policy updates run on the same devices. Only the language-model weights are trained, and they are trained through LoRA adapters on all linear projections, so a single experiment fits on one 80 GB GPU. The vision-language model is never trained: a frozen Qwen3.5-27B is served as an HTTP endpoint and reached only through the agent's `batch_look` perception tool, identically during trajectory collection, training rollouts, and evaluation. Because both collection and evaluation drive the same CodeAct agent loop, the policy that is trained is exactly the policy that is evaluated.

### 6.2 Data

Three corpora play distinct roles. Supervised fine-tuning uses transfer trajectories collected on MMLongBench-Doc [MMLongBench-Doc, arXiv:2407.01523], a multi-page document-understanding corpus disjoint from the DocVQA-2026 validation set, so SFT supervision carries no leakage into the evaluation split. Evaluation uses the DocVQA-2026 validation set (25 documents, 80 questions); the benchmark provides no training split, and its test set is held out by the competition portal. The DocVQA-family training-data pool described in §5.3 is a leakage-safe prompt substrate assembled for the upcoming larger-data SFT and RL/OPD runs; it is not a source of any number reported here, and at the time of these experiments no model has been trained on it.

### 6.3 Evaluation protocol

The metric is binary ANLS at a threshold of 0.9, the official DocVQA-2026 scoring rule introduced in §2: a per-question ANLS at or above 0.9 counts as correct and anything below as wrong, and the reported score is the resulting thresholded accuracy averaged over questions. The same metric definition governs every number in this report and serves as the GRPO reward in §5.

Each question is answered with $n{=}4$ independent samples to reduce the per-question variance of stochastic rollouts. Two evaluation scales are used: a 29-question mini-screen for rapid checkpoint comparison and the full 80-question validation set for confirmation. Comparisons between a model and its baseline are made per question on the same items, so improvements are assessed with paired statistics rather than unpaired aggregate scores, and reported $p$-values are paired tests over the shared question set.

Three reference points anchor the trained-model results: the untrained Qwen3.5-4B agent in the same CodeAct scaffold (the policy's own zero-shot self, §7.1), the published ≤8B DocVQA leaderboard entry at roughly 0.19 ANLS, and the 8B–35B-tier entry at roughly 0.375.

### 6.4 Thinking setting across pillars

The two pillars run the reasoner under different native-thinking settings, which the comparisons below take into account. The Pillar-A harness evaluation of the 27B reasoner (§3) is run with native thinking disabled (`enable_thinking=false`): at 27B a thinking ablation on the strongest scaffold yields no gain and is prone to generation hangs. The Pillar-B training and evaluation of the 4B agent (§5–§7) are run with native thinking enabled. The CodeAct training agent loop exposes no separate reasoning field, so native thinking is the agent's only reasoning channel; disabling it would remove reasoning from the policy that is trained and deployed. The two settings are therefore a deliberate, documented protocol difference between a frozen-reasoner harness study and a trainable-reasoner study, not a discrepancy between otherwise-identical runs. Cross-pillar numbers are compared with this difference in view.

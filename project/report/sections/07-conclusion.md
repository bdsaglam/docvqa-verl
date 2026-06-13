## 7 Conclusion and Future Work

Across two model families and four reasoner scales, what governs document-agent
accuracy on long, visually dense documents is not the size of the reasoner but
whether it can actively control perception: an agent that writes code to crop,
zoom, and query a frozen vision–language model on demand matches or exceeds
much larger frozen models and far surpasses fixed-granularity tool agents. The
advantage is mechanistically located — it concentrates where cropping recovers
detail a whole-page read misses, collapses when visual perception is replaced by
optical character recognition, and appears only once the base model is capable
enough, in both reasoning and code, to drive the loop.

The two strongest scaffolds in this family tie at the 27B backbone, and the
append-only member preserves the growing-prefix trajectory that established
weight-level training assumes; it is therefore the principled target for training
a small agent. A preliminary study at the ≤8B tier finds that supervised
fine-tuning buys a small, fragile lift and that the dominant obstacle is
structural: on long documents the append-only trajectory grows until the agent
exhausts its budget before answering. Training a small agent to realize the
active-perception advantage — past that runaway ceiling, with on-policy methods
that match the deployment distribution — is the open direction this work points
to.

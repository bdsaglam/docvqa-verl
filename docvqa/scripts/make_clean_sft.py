"""Build a clean SeqKD SFT parquet from an eval.py run-dir.

Filters to anls==1.0 + termination==submit trajectories and cleans each assistant
turn to a single `<reasoning>...</think>\\n\\n```python ... ```` block:

- **keep-first-fence** truncation drops anything after the first code fence — this
  removes the 27B's role-played *next-turn* tails (stray `</think>` + extra code
  blocks). Because the live scaffold runs `parse_first_fence=True`, the FIRST fence
  is what executed, so its observation stays consistent after truncation.
- NOTE the boundary uses the FIRST `</think>` (the real reasoning end), then the
  first code fence after it. (An earlier inline version used `rfind` = the *last*
  `</think>`, which kept the stray tail — fixed here.)

Usage:
  python docvqa/scripts/make_clean_sft.py --in outputs/runs/mmlb-collect-v4 \
      --out data/sft/clean_v5.parquet --max-per-question 3 --max-tokens 12000
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n.*?\n```", re.DOTALL)


def keep_first_fence(content: str) -> str:
    """Keep `<reasoning>...</think>` + the first code fence; drop any role-played tail."""
    t = content.find("</think>")
    start = t + len("</think>") if t >= 0 else 0
    m = _FENCE.search(content, start)
    return content[: m.end()] if m else content


def _approx_tokens(messages: list[dict], tok) -> int:
    return sum(len(tok.encode(str(m.get("content") or ""), add_special_tokens=False)) + 8 for m in messages)


def _read_run_dir(path: Path):
    files = sorted(path.glob("tasks/*/trajectories.jsonl")) if path.is_dir() else [path]
    for fp in files:
        for line in fp.open():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue  # robust to a partial last line while collection still writing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-question", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    by_q: dict[str, list] = defaultdict(list)
    n_seen = n_truncated = 0
    for r in _read_run_dir(Path(args.inp)):
        if r.get("anls") != 1.0 or r.get("termination") != "submit":
            continue
        msgs = r.get("messages") or []
        if not msgs or not any(m.get("role") == "assistant" for m in msgs):
            continue
        n_seen += 1
        clean = []
        for m in msgs:
            if m.get("role") == "assistant":
                nc = keep_first_fence(m.get("content", ""))
                if nc != m.get("content", ""):
                    n_truncated += 1
                clean.append({"role": "assistant", "content": nc})
            else:
                clean.append({"role": m["role"], "content": m.get("content", "")})
        if _approx_tokens(clean, tok) > args.max_tokens:
            continue
        by_q[r.get("question_id", r.get("record_id", ""))].append(clean)

    kept = []
    for _, lst in by_q.items():
        kept.extend(lst[: args.max_per_question])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"messages": kept}).to_parquet(out)
    # sanity: zero stray-think turns
    stray = sum(
        1 for k in kept for m in k if m["role"] == "assistant" and m["content"].count("</think>") > 1
    )
    print(
        f"clean SFT: {len(kept)} traj / {len(by_q)} Q (from {n_seen} anls=1.0; "
        f"{n_truncated} turns first-fence-truncated; stray-</think> turns remaining={stray}) -> {out}"
    )


if __name__ == "__main__":
    main()

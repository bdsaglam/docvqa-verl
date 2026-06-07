"""Trajectory viewer — a small FastAPI app to inspect eval-run trajectories.

Reads the on-disk layout produced by `docvqa/scripts/eval.py`:

    <runs-dir>/<run>/config.json
    <runs-dir>/<run>/results.json
    <runs-dir>/<run>/tasks/<doc_id>/result.json
    <runs-dir>/<run>/tasks/<doc_id>/trajectories.jsonl   # one rollout per line

No DB, no preprocessing: every endpoint reads the filesystem on request, so
new runs/tasks show up on refresh. Run with the repo venv:

    .venv/bin/python -m uvicorn tools.trajviewer.app:app --port 8765
    # or: .venv/bin/python tools/trajviewer/app.py --port 8765 --runs-dir outputs/runs
"""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = Path(os.environ.get("TRAJVIEWER_RUNS_DIR", REPO_ROOT / "outputs" / "runs"))
DATA_DIR = Path(os.environ.get("TRAJVIEWER_DATA_DIR", REPO_ROOT / "data"))

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="DocVQA Trajectory Viewer")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _read_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _safe_run_dir(run: str) -> Path:
    # prevent path traversal; runs are single path components
    if "/" in run or "\\" in run or run in ("", ".", ".."):
        raise HTTPException(400, "bad run name")
    d = RUNS_DIR / run
    if not d.is_dir():
        raise HTTPException(404, f"run not found: {run}")
    return d


def _iter_rollouts(run_dir: Path):
    """Yield (doc_id, parsed-rollout-dict) for every trajectory line in a run."""
    tasks = run_dir / "tasks"
    if not tasks.is_dir():
        return
    for doc_dir in sorted(tasks.iterdir()):
        traj = doc_dir / "trajectories.jsonl"
        if not traj.is_file():
            continue
        with open(traj) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield doc_dir.name, json.loads(line)
                except json.JSONDecodeError:
                    continue


def _summarize_rollout(d: dict) -> dict:
    """Triage-row fields only (no heavy messages/token-ids payload)."""
    return {
        "doc_id": d.get("doc_id"),
        "category": d.get("category"),
        "question_id": d.get("question_id"),
        "sample_idx": d.get("sample_idx", 0),
        "question": d.get("question"),
        "gold_answer": d.get("gold_answer"),
        "submitted_answer": d.get("submitted_answer"),
        "extracted_answer": d.get("extracted_answer"),
        "is_correct": d.get("is_correct"),
        "anls": d.get("anls"),
        "termination": d.get("termination"),
        "num_turns": d.get("num_turns"),
        "vlm_calls": d.get("vlm_calls"),
        "wall_clock_s": d.get("wall_clock_s"),
    }


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/api/runs")
def list_runs():
    runs = []
    if not RUNS_DIR.is_dir():
        return {"runs_dir": str(RUNS_DIR), "runs": []}
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        cfg = _read_json(d / "config.json") or {}
        res = _read_json(d / "results.json") or {}
        summary = res.get("summary", {})
        n_tasks = len(list((d / "tasks").iterdir())) if (d / "tasks").is_dir() else 0
        runs.append({
            "run": d.name,
            "created_at": cfg.get("created_at"),
            "dataset": cfg.get("dataset"),
            "split": cfg.get("split"),
            "model": cfg.get("model"),
            "n": cfg.get("n"),
            "num_questions": cfg.get("num_questions"),
            "n_tasks": n_tasks,
            "overall_accuracy": summary.get("overall_accuracy"),
            "total_questions": summary.get("total_questions"),
            "has_results": (d / "results.json").is_file(),
            "mtime": d.stat().st_mtime,
        })
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return {"runs_dir": str(RUNS_DIR), "runs": runs}


@app.get("/api/runs/{run}")
def run_detail(run: str):
    d = _safe_run_dir(run)
    return {
        "run": run,
        "config": _read_json(d / "config.json"),
        "results": _read_json(d / "results.json"),
    }


@app.get("/api/runs/{run}/rollouts")
def run_rollouts(run: str):
    d = _safe_run_dir(run)
    rows = [_summarize_rollout(r) for _, r in _iter_rollouts(d)]
    return {"run": run, "rollouts": rows}


@app.get("/api/runs/{run}/tasks/{doc_id}/{question_id}/{sample_idx}")
def rollout_detail(run: str, doc_id: str, question_id: str, sample_idx: int):
    d = _safe_run_dir(run)
    traj = d / "tasks" / doc_id / "trajectories.jsonl"
    if not traj.is_file():
        raise HTTPException(404, "task not found")
    with open(traj) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("question_id") == question_id and rec.get("sample_idx", 0) == sample_idx:
                # drop the bulky token-id arrays; keep lengths for reference
                rec.pop("prompt_ids", None)
                rec.pop("response_ids", None)
                rec.pop("response_mask", None)
                return rec
    raise HTTPException(404, "rollout not found")


@lru_cache(maxsize=512)
def _doc_pages_dir(dataset: str, split: str, doc_id: str) -> str | None:
    cand = DATA_DIR / dataset / split / "docs" / doc_id / "pages"
    return str(cand) if cand.is_dir() else None


@app.get("/api/runs/{run}/doc/{doc_id}/pages")
def doc_pages(run: str, doc_id: str):
    """List available page image filenames for a doc (sorted by page number)."""
    d = _safe_run_dir(run)
    cfg = _read_json(d / "config.json") or {}
    dataset, split = cfg.get("dataset"), cfg.get("split")
    if not dataset or not split:
        return {"pages": [], "dataset": dataset, "split": split}
    pdir = _doc_pages_dir(dataset, split, doc_id)
    if not pdir:
        return {"pages": [], "dataset": dataset, "split": split}
    pages = []
    for p in Path(pdir).glob("page_*.png"):
        try:
            n = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        pages.append(n)
    pages.sort()
    return {"pages": pages, "dataset": dataset, "split": split}


@app.get("/api/runs/{run}/doc/{doc_id}/page/{page}")
def doc_page_image(run: str, doc_id: str, page: int):
    d = _safe_run_dir(run)
    cfg = _read_json(d / "config.json") or {}
    dataset, split = cfg.get("dataset"), cfg.get("split")
    pdir = _doc_pages_dir(dataset or "", split or "", doc_id)
    if not pdir:
        raise HTTPException(404, "pages dir not found")
    img = Path(pdir) / f"page_{page}.png"
    if not img.is_file():
        raise HTTPException(404, "page not found")
    return FileResponse(img, media_type="image/png")


# Static SPA (mounted last so /api/* wins)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main():
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--runs-dir", default=None, help="dir containing <run>/ subdirs")
    ap.add_argument("--data-dir", default=None, help="repo data/ root for page images")
    args = ap.parse_args()

    global RUNS_DIR, DATA_DIR
    if args.runs_dir:
        RUNS_DIR = Path(args.runs_dir).resolve()
    if args.data_dir:
        DATA_DIR = Path(args.data_dir).resolve()
    print(f"runs-dir: {RUNS_DIR}")
    print(f"data-dir: {DATA_DIR}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

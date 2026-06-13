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
    """Triage-row fields only (no heavy messages/token-ids payload). Token
    lengths are cheap here (the arrays are already parsed) and are the key
    distribution for SFT sequence-length budgeting."""
    pid, rid = d.get("prompt_ids"), d.get("response_ids")
    plen = len(pid) if isinstance(pid, list) else None
    rlen = len(rid) if isinstance(rid, list) else None
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
        "prompt_tokens": plen,
        "response_tokens": rlen,
        "num_tokens": (plen + rlen) if (plen is not None and rlen is not None) else None,
        "num_pages": None,  # filled in by the rollouts endpoint (per-doc join)
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


@lru_cache(maxsize=64)
def _run_doc_pagecounts(questions_path: str, _mtime: float) -> dict:
    """doc_id -> num_pages, from each doc's metadata.json (fallback: count
    page_*.png). Cached per questions file; _mtime busts on change."""
    data = _read_json(Path(questions_path)) or []
    records = data if isinstance(data, list) else list(data.values())
    out = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        did, dd = r.get("doc_id"), r.get("doc_dir")
        if not did or did in out or not dd:
            continue
        md = _read_json(Path(dd) / "metadata.json")
        if md and isinstance(md.get("num_pages"), int):
            out[did] = md["num_pages"]
        else:
            pdir = Path(dd) / "pages"
            if pdir.is_dir():
                out[did] = sum(1 for _ in pdir.glob("page_*.png"))
    return out


def _run_pagecount_map(run_dir: Path) -> dict:
    cfg = _read_json(run_dir / "config.json") or {}
    qpath = cfg.get("questions")
    if not qpath:
        return {}
    qp = Path(qpath)
    if not qp.is_absolute():
        qp = REPO_ROOT / qp
    if not qp.is_file():
        return {}
    return _run_doc_pagecounts(str(qp), qp.stat().st_mtime)


@app.get("/api/runs/{run}/rollouts")
def run_rollouts(run: str):
    d = _safe_run_dir(run)
    pages = _run_pagecount_map(d)
    rows = []
    for doc_id, r in _iter_rollouts(d):
        row = _summarize_rollout(r)
        if row.get("num_pages") is None:
            row["num_pages"] = pages.get(doc_id)
        rows.append(row)
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


@lru_cache(maxsize=64)
def _questions_doc_map(questions_path: str, _mtime: float) -> dict:
    """doc_id -> doc_dir, parsed from a run's questions JSON (the authoritative
    source of where a doc's pages live). _mtime busts the cache on file change."""
    data = _read_json(Path(questions_path))
    if data is None:
        return {}
    records = data if isinstance(data, list) else list(data.values())
    out = {}
    for r in records:
        if isinstance(r, dict) and r.get("doc_id") and r.get("doc_dir"):
            out[r["doc_id"]] = r["doc_dir"]
    return out


def _resolve_pages_dir(run_dir: Path, doc_id: str) -> Path | None:
    """Find a doc's pages/ dir. Prefer the questions file's per-doc `doc_dir`
    (config's dataset/split fields are often wrong/missing); fall back to the
    DATA_DIR/<dataset>/<split>/docs/<doc_id> guess."""
    cfg = _read_json(run_dir / "config.json") or {}
    qpath = cfg.get("questions")
    if qpath:
        qp = Path(qpath)
        if not qp.is_absolute():
            qp = REPO_ROOT / qp
        if qp.is_file():
            dd = _questions_doc_map(str(qp), qp.stat().st_mtime).get(doc_id)
            if dd:
                pdir = Path(dd) / "pages"
                if pdir.is_dir():
                    return pdir
    dataset, split = cfg.get("dataset"), cfg.get("split")
    if dataset and split:
        cand = DATA_DIR / dataset / split / "docs" / doc_id / "pages"
        if cand.is_dir():
            return cand
    return None


@app.get("/api/runs/{run}/doc/{doc_id}/pages")
def doc_pages(run: str, doc_id: str):
    """List available page image filenames for a doc (sorted by page number)."""
    d = _safe_run_dir(run)
    pdir = _resolve_pages_dir(d, doc_id)
    if not pdir:
        return {"pages": [], "resolved": None}
    pages = []
    for p in pdir.glob("page_*.png"):
        try:
            n = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        pages.append(n)
    pages.sort()
    return {"pages": pages, "resolved": str(pdir)}


@app.get("/api/runs/{run}/doc/{doc_id}/page/{page}")
def doc_page_image(run: str, doc_id: str, page: int):
    d = _safe_run_dir(run)
    pdir = _resolve_pages_dir(d, doc_id)
    if not pdir:
        raise HTTPException(404, "pages dir not found")
    img = pdir / f"page_{page}.png"
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

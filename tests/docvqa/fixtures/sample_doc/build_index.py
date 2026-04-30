"""Build BM25 index for sample_doc fixture. Run once, committed output."""
import json
from pathlib import Path
import bm25s
import Stemmer

ROOT = Path(__file__).parent
ocr_dir = ROOT / "ocr"
bm25_dir = ROOT / "bm25"
bm25_dir.mkdir(exist_ok=True)

chunks = []
for p in sorted(ocr_dir.glob("page_*.md")):
    page = int(p.stem.split("_")[1])
    chunks.append({"page": page, "text": p.read_text()})

(bm25_dir / "chunks.json").write_text(json.dumps(chunks, indent=2))

corpus = [c["text"] for c in chunks]
tokens = bm25s.tokenize(corpus, stemmer=Stemmer.Stemmer("english"))
retriever = bm25s.BM25()
retriever.index(tokens)
retriever.save(bm25_dir, corpus=None)
print(f"Wrote {len(chunks)} chunks to {bm25_dir}")

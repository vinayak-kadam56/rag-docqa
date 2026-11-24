"""Measure retrieval quality on a small labelled question set.

    python scripts/evaluate.py                       # uses EMBEDDINGS env var (default: auto)
    EMBEDDINGS=hash python scripts/evaluate.py       # force the offline hashing embedder

Each question in eval/questions.json names the document and a keyword that the
retrieved chunk must contain. Reported metrics:

  hit@1 / hit@3  - was a chunk from the right document ranked 1st / within the top 3?
  MRR            - mean reciprocal rank of the first correct document (1.0 = always first)
  keyword@3      - did a top-3 chunk from the right document actually contain the answer text?
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embeddings import get_embedder          # noqa: E402
from app.rag import RAGService                   # noqa: E402
from app.vectorstore import VectorStore          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "eval"


def main():
    embedder = get_embedder()
    questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmp:
        service = RAGService(VectorStore(tmp, embedder.name), embedder, llm=None)
        for path in sorted((ROOT / "docs").glob("*.txt")):
            service.ingest(path.name, path.read_bytes())

        hit1 = hit3 = kw3 = 0
        reciprocal = []
        for q in questions:
            hits = service.retrieve(q["question"], k=5)
            ranks = [i for i, h in enumerate(hits, 1) if h["filename"] == q["doc"]]
            first = ranks[0] if ranks else None
            hit1 += first == 1
            hit3 += first is not None and first <= 3
            reciprocal.append(1 / first if first else 0.0)
            kw3 += any(h["filename"] == q["doc"] and q["keyword"].lower() in h["text"].lower()
                       for h in hits[:3])
            status = "OK  " if first == 1 else "MISS"
            print(f"{status} rank={first}  {q['question']}")

    n = len(questions)
    print(f"\nembedder: {embedder.name}   questions: {n}")
    print(f"hit@1: {hit1 / n:.0%}   hit@3: {hit3 / n:.0%}   MRR: {sum(reciprocal) / n:.2f}   keyword@3: {kw3 / n:.0%}")


if __name__ == "__main__":
    main()

"""The RAG pipeline: ingest (load -> chunk -> embed -> store) and ask (retrieve -> generate)."""
import os

from app.chunking import chunk_text
from app.embeddings import get_embedder
from app.llm import SYSTEM_PROMPT, build_user_prompt, get_llm
from app.loaders import extract_text
from app.vectorstore import VectorStore

NOT_FOUND = "I couldn't find anything relevant to that question in the uploaded documents."


class RAGService:
    def __init__(self, store, embedder, llm=None, chunk_size=120, overlap=30,
                 top_k=4, min_score=0.05):
        self.store, self.embedder, self.llm = store, embedder, llm
        self.chunk_size, self.overlap = chunk_size, overlap
        self.top_k, self.min_score = top_k, min_score

    def ingest(self, filename: str, data: bytes) -> dict:
        text = extract_text(filename, data)
        chunks = chunk_text(text, self.chunk_size, self.overlap)
        if not chunks:
            raise ValueError("No readable text found in this file.")
        doc_id = self.store.add_document(filename, chunks, self.embedder.embed(chunks))
        return {"doc_id": doc_id, "filename": filename, "chunks": len(chunks)}

    def retrieve(self, question: str, k=None) -> list[dict]:
        query_vector = self.embedder.embed([question])[0]
        return self.store.search(query_vector, k or self.top_k)

    def ask(self, question: str, k=None) -> dict:
        hits = [h for h in self.retrieve(question, k) if h["score"] >= self.min_score]
        provider = self.llm.name if self.llm else "none"
        if not hits:
            return {"answer": NOT_FOUND, "sources": [], "provider": provider}

        if self.llm is None:   # no LLM configured: show the best passage instead of generating
            answer = f"Best matching passage (from {hits[0]['filename']}):\n\n{hits[0]['text']}"
        else:
            answer = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, hits))

        sources = [
            {"filename": h["filename"], "chunk_index": h["chunk_index"],
             "score": round(h["score"], 3), "text": h["text"]}
            for h in hits
        ]
        return {"answer": answer, "sources": sources, "provider": provider}


def build_service() -> RAGService:
    embedder = get_embedder()
    store = VectorStore(os.getenv("DATA_DIR", "data"), embedder.name)
    return RAGService(
        store, embedder, get_llm(),
        chunk_size=int(os.getenv("CHUNK_SIZE", "120")),
        overlap=int(os.getenv("CHUNK_OVERLAP", "30")),
        top_k=int(os.getenv("TOP_K", "4")),
        min_score=float(os.getenv("MIN_SCORE", "0.05")),
    )

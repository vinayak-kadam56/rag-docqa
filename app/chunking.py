"""Split text into overlapping word-based chunks."""


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    """Overlap keeps a sentence that straddles a boundary retrievable from both chunks."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be > 0 and 0 <= overlap < chunk_size")
    words = text.split()
    step = chunk_size - overlap
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks

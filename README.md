# Document Q&A with Retrieval-Augmented Generation (RAG)

Upload PDFs / text files, then ask questions. The app finds the most relevant passages
with **vector similarity search** and (optionally) lets an LLM write an answer grounded
**only** in those passages, with numbered source citations.

```
upload -> extract text -> chunk (overlap) -> embed -> vector store
question -> embed -> cosine top-k search -> prompt with passages -> LLM answer + sources
```

## Features

- PDF, `.txt` and `.md` ingestion; overlapping word-based chunking
- Swappable **embedders**: `sentence-transformers` (semantic) or a zero-download hashing embedder
- Own persistent **vector store** (numpy, exact cosine search, atomic saves)
- Swappable **LLMs**: Claude (Anthropic API), a local model via Ollama, or no LLM at all
  (returns the best matching passage, so the project works with no API key)
- Grounded answers with citations; irrelevant questions are refused instead of hallucinated
- Web UI (`/`) and API docs (`/docs`); upload size limit; XSS-safe rendering
- **Retrieval evaluation script** (hit@k, MRR) and 25 unit tests + API tests
- Dockerfile and Docker Compose

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate             # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Open <http://localhost:8000>. Out of the box it uses the offline embedder and returns the
best matching passage.

### Better quality (optional)

```bash
pip install -r requirements-extra.txt         # sentence-transformers + anthropic

# semantic embeddings (downloads a ~90 MB model on first run)
set EMBEDDINGS=sbert                           # Linux/macOS: export EMBEDDINGS=sbert

# LLM answers - choose one
set LLM_PROVIDER=anthropic & set ANTHROPIC_API_KEY=your-key      # Claude
set LLM_PROVIDER=ollama    & set LLM_MODEL=llama3.2              # local, free (needs Ollama)
```

If you change the embedder, delete the `data/` folder - vectors from different embedders
are not compatible (the app refuses to load a mismatched index).

## Run with Docker

```bash
docker compose up --build
# with extras:  INSTALL_EXTRAS=true EMBEDDINGS=sbert docker compose up --build
```

## API

| Method | Path                       | Description                          |
|--------|----------------------------|--------------------------------------|
| POST   | `/api/documents`           | Upload a PDF / txt / md (multipart)  |
| GET    | `/api/documents`           | List indexed documents               |
| DELETE | `/api/documents/{doc_id}`  | Remove a document                    |
| POST   | `/api/ask`                 | `{"question": "...", "top_k": 4}`    |
| GET    | `/health`                  | Active embedder and LLM              |

## Evaluation

```bash
python scripts/evaluate.py                    # uses your configured embedder
EMBEDDINGS=hash python scripts/evaluate.py    # offline baseline
```

Reports hit@1, hit@3, MRR and keyword@3 on `eval/questions.json` (10 labelled questions
over 3 documents). The set is small and easy - add your own documents and harder questions
(paraphrased wording, multi-document questions) to get numbers that mean something, and
compare `hash` vs `sbert`.

## Design decisions

- **Own vector store instead of a database.** Exact cosine search over a numpy matrix is
  simple, correct, and fast enough for thousands of chunks. Beyond that, swap in an ANN
  index (FAISS/HNSW) or a vector DB; the store is isolated in `app/vectorstore.py`.
- **Chunk overlap** so an answer that straddles a boundary is still retrievable.
- **Relevance threshold (`MIN_SCORE`)**: below it the LLM is never called and the app says
  it found nothing, which reduces hallucinated answers and saves API cost.
- **System prompt restricts the model to the retrieved context** and asks for citations.
- **Index/embedder compatibility check** prevents silently mixing incompatible vectors.

## Limitations / future work

- Word-count chunking ignores document structure (headings, paragraphs); semantic or
  structure-aware chunking would help.
- No reranking step (a cross-encoder over the top-20 would improve precision).
- Scanned PDFs (images) need OCR; text extraction returns nothing for them.
- Single-user, no authentication.

"""Document Q&A (RAG) API - upload documents, then ask questions about them."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.llm import LLMError
from app.rag import build_service

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = build_service()   # reads config from environment variables
    yield


app = FastAPI(
    title="Document Q&A (RAG)",
    description="Upload PDFs / text files and ask questions answered from their content.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(request: Request):
    svc = request.app.state.service
    return {"status": "ok", "embedder": svc.embedder.name, "llm": svc.llm.name if svc.llm else "none"}


@app.post("/api/documents", status_code=201)
def upload_document(request: Request, file: UploadFile = File(...)):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES // 1024 // 1024} MB).")
    filename = os.path.basename(file.filename or "upload")
    try:
        return request.app.state.service.ingest(filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/documents")
def list_documents(request: Request):
    return request.app.state.service.store.list_documents()


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, request: Request):
    if not request.app.state.service.store.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": doc_id}


@app.post("/api/ask")
def ask(body: AskRequest, request: Request):
    try:
        return request.app.state.service.ask(body.question, body.top_k)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

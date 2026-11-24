"""Turn an uploaded file into clean text."""
import io
import re
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Upload a .txt, .md or .pdf file.")

    if ext == ".pdf":
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:  # corrupt / encrypted PDFs raise many different errors
            raise ValueError("Could not read this PDF (corrupt or password-protected?).") from exc
    else:
        text = data.decode("utf-8", errors="replace")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

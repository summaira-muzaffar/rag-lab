"""STEP 1 - LOAD: read files from disk into Document objects."""
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Document:
    text: str
    source: str       # file name, used later for citations
    page: int = 0     # PDF page number (0 for text files)


def load_documents(docs_dir: str) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(Path(docs_dir).rglob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            docs.append(Document(path.read_text(encoding="utf-8"), path.name))
        elif path.suffix.lower() == ".pdf":
            for i, page in enumerate(PdfReader(str(path)).pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(text, path.name, i))
    return docs

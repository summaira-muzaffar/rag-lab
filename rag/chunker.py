"""STEP 2 - CHUNK: split documents into small overlapping pieces.

Why: the retriever finds focused pieces, and the LLM only needs the relevant part.
Overlap keeps a sentence that sits on a boundary from losing its context.
"""
import re
from dataclasses import dataclass, field

from .loader import Document


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _units(text: str, size: int):
    """Yield paragraphs; break long paragraphs into sentences; hard-cut very long sentences."""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            yield para
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            while len(sent) > size:
                yield sent[:size]
                sent = sent[size:]
            if sent:
                yield sent


def split_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    chunks: list[str] = []
    cur = ""
    for unit in _units(text, size):
        if cur and len(cur) + len(unit) + 1 > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            if " " in tail:
                tail = tail.split(" ", 1)[1]  # start the overlap on a whole word
            cur = f"{tail} {unit}".strip()
        else:
            cur = f"{cur}\n{unit}".strip() if cur else unit
    if cur:
        chunks.append(cur)
    return chunks


def chunk_documents(docs: list[Document], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        for i, piece in enumerate(split_text(doc.text, size, overlap)):
            chunks.append(
                Chunk(
                    id=f"{doc.source}::p{doc.page}::c{i}",
                    text=piece,
                    metadata={"source": doc.source, "page": doc.page, "chunk": i},
                )
            )
    return chunks

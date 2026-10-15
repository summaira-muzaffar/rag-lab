"""Glue: the whole RAG flow in one class."""
from dataclasses import dataclass

from config import settings

from .chunker import chunk_documents
from .embedder import get_embedder
from .llm import get_llm
from .loader import load_documents
from .retriever import Retriever
from .store import VectorStore


@dataclass
class Answer:
    text: str
    sources: list[dict]


class RAG:
    def __init__(self, with_llm: bool = True):
        self.embedder = get_embedder()
        self.store = VectorStore()
        self.retriever = Retriever(self.embedder, self.store, settings.top_k)
        self.llm = get_llm() if with_llm else None

    def ingest(self, reset: bool = True) -> int:
        """load -> chunk -> embed -> store"""
        if reset:
            self.store.reset()
        docs = load_documents(settings.docs_dir)
        chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
        for i in range(0, len(chunks), 64):  # batches keep memory small
            batch = chunks[i:i + 64]
            self.store.add(batch, self.embedder.embed([c.text for c in batch]))
        return len(chunks)

    def ask(self, question: str) -> Answer:
        """retrieve -> generate"""
        hits = self.retriever.retrieve(question)
        return Answer(self.llm.answer(question, hits), hits)

"""STEP 4 - STORE: a vector database (ChromaDB) that saves vectors on disk and searches them."""
import chromadb

from config import settings


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_dir)
        self.col = self.client.get_or_create_collection(
            settings.collection, metadata={"hnsw:space": "cosine"}
        )

    def reset(self):
        self.client.delete_collection(settings.collection)
        self.col = self.client.get_or_create_collection(
            settings.collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks, embeddings):
        self.col.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            embeddings=embeddings,
        )

    def search(self, query_embedding, k: int):
        res = self.col.query(query_embeddings=[query_embedding], n_results=k)
        hits = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append({"text": text, "source": meta["source"], "page": meta["page"],
                         "score": round(1 - dist, 3)})   # cosine similarity
        return hits

    def count(self) -> int:
        return self.col.count()

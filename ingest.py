"""Run once (and again whenever docs change):  python ingest.py"""
from rag.pipeline import RAG

if __name__ == "__main__":
    n = RAG(with_llm=False).ingest()
    print(f"Ingested {n} chunks.")

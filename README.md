# RAG Lab: plain RAG + LLM

## Architecture

```
INGEST (run once)                              ASK (every question)
data/docs/*.md|txt|pdf                         user question
   |  loader.py   read files                      |  embedder.py   question -> vector
   v                                              v
 Documents                                     store.py    find top-k similar chunks
   |  chunker.py  split + overlap                 |  retriever.py
   v                                              v
 Chunks                                        llm.py      prompt = context + question
   |  embedder.py text -> vectors                 |         LLM answers + cites [1] [2]
   v                                              v
 store.py  ChromaDB (saved in chroma_db/)      Answer + sources
```

| File | Job |
|---|---|
| `config.py` | All settings (chunk size, top_k, models) |
| `rag/loader.py` | Read .md / .txt / .pdf |
| `rag/chunker.py` | Split text into overlapping chunks |
| `rag/embedder.py` | Text to vectors (local model, or offline test mode) |
| `rag/store.py` | ChromaDB: save and search vectors |
| `rag/retriever.py` | Question to top-k chunks |
| `rag/llm.py` | Prompt + Claude call (or echo mode) |
| `rag/pipeline.py` | Connects everything: `ingest()` and `ask()` |
| `ingest.py` / `ask.py` / `api.py` / `evaluate.py` | Entry points |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # add your ANTHROPIC_API_KEY

python ingest.py                  # build the vector DB (first run downloads the embedding model)
python ask.py "How many sick days do I get?"
python evaluate.py                # retrieval score on eval/questions.json
uvicorn api:app --reload          # optional web API at /docs
```

No API key yet? Set `LLM_BACKEND=echo` to see only the retrieved text.

## Put your own documents in

Drop PDFs or text files into `data/docs/`, rerun `python ingest.py`, and replace `eval/questions.json`
with 20-30 questions about your files.

## Experiments to learn from (change one thing at a time, rerun evaluate.py)

1. `CHUNK_SIZE`: 200 vs 500 vs 1000
2. `TOP_K`: 1, 2, 4, 8
3. `CHUNK_OVERLAP`: 0 vs 80
4. Ask something the docs don't cover and check you get "I don't know based on the documents."

## Next upgrades (still plain RAG)

Hybrid search (BM25 + vectors), a reranker, metadata filters, query rewriting, RAGAS evaluation, Streamlit UI, Docker.

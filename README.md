# RAG Lab

Ask questions about your own documents. The app finds the most relevant passages, gives them to Gemini, and answers with citations like `[1]`.

## Quick start

| Step | Windows | Mac / Linux |
|---|---|---|
| 1. Create environment | `python -m venv .venv` | `python -m venv .venv` |
| 2. Activate it | `.venv\Scripts\activate` | `source .venv/bin/activate` |
| 3. Install | `pip install -r requirements.txt` | `pip install -r requirements.txt` |
| 4. Create settings file | `copy .env.example .env` | `cp .env.example .env` |
| 5. Add your key | Open `.env`, set `GEMINI_API_KEY=...` | same |
| 6. Build the index | `python ingest.py` | `python ingest.py` |
| 7. Ask | `python ask.py "How many sick days do I get?"` | same |

Get a free key at [aistudio.google.com](https://aistudio.google.com) > **Get API key**.
The first `ingest.py` run downloads a small embedding model, so it takes a minute.

No key yet? Set `LLM_BACKEND=echo` in `.env` to see only the retrieved text.

## Ways to run it

| Want | Command |
|---|---|
| Terminal Q&A | `python ask.py` |
| Web app (chat UI) | `streamlit run app.py --server.fileWatcherType none` |
| Web API (docs at `/docs`) | `uvicorn api:app --reload` |
| Retrieval score | `python evaluate.py` |

In the web app, click **Build / rebuild index** in the sidebar once before asking.

## Use your own documents

1. Put `.md`, `.txt` or `.pdf` files in `data/docs/`.
2. Run `python ingest.py` again.
3. Replace `eval/questions.json` with 20-30 questions about your files.

## Settings (`.env`)

| Setting | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | none | Your Gemini key. Never commit `.env`. |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Model used to write answers |
| `LLM_BACKEND` | `gemini` | `gemini` or `echo` (no LLM) |
| `EMBED_BACKEND` | `local` | `local` (real model) or `hash` (offline test) |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `80` | Characters shared between chunks |
| `TOP_K` | `4` | Passages sent to the LLM |

## How it works

```
INGEST (once)                         ASK (every question)
documents -> chunks -> vectors        question -> vector -> top-k chunks
          -> saved in chroma_db/                         -> Gemini -> answer + sources
```

| File | Job |
|---|---|
| `config.py` | All settings |
| `rag/loader.py` | Read .md / .txt / .pdf |
| `rag/chunker.py` | Split text into overlapping chunks |
| `rag/embedder.py` | Text to vectors |
| `rag/store.py` | ChromaDB: save and search vectors |
| `rag/retriever.py` | Question to top-k chunks |
| `rag/llm.py` | Prompt + Gemini call (or echo mode) |
| `rag/pipeline.py` | Connects everything |
| `ingest.py`, `ask.py`, `evaluate.py` | Command-line tools |
| `app.py` | Streamlit chat app |
| `api.py` | FastAPI web API |

## Troubleshooting

| Problem | Fix |
|---|---|
| `API key not valid` | Check `.env`: no quotes, no spaces around `=` |
| `429` error | Free tier limit. Wait a minute and retry |
| Answer is "I don't know based on the documents" | The documents don't cover it. This is correct behavior |
| No results or empty answers | Run `python ingest.py` again |

## Experiments

Change one setting at a time, then re-run `python ingest.py` and `python evaluate.py`.

| Setting | Try |
|---|---|
| `CHUNK_SIZE` | 200, 500, 1000 |
| `TOP_K` | 1, 2, 4, 8 |
| `CHUNK_OVERLAP` | 0, 80 |

Next upgrades: hybrid search (BM25 + vectors), a reranker, metadata filters, query rewriting, RAGAS evaluation, Docker.

"""All settings in one place. Override any of them with environment variables or a .env file."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    docs_dir: str = os.getenv("DOCS_DIR", "data/docs")
    chroma_dir: str = os.getenv("CHROMA_DIR", "chroma_db")
    collection: str = os.getenv("COLLECTION", "docs")

    embed_backend: str = os.getenv("EMBED_BACKEND", "local")  # local | hash
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))      # characters
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))  # characters
    top_k: int = int(os.getenv("TOP_K", "4"))

    llm_backend: str = os.getenv("LLM_BACKEND", "anthropic")   # anthropic | echo
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-5-5")


settings = Settings()

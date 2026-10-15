"""STEP 3 - EMBED: turn text into vectors so 'similar meaning' = 'close together'."""
import math
import re
import zlib

from config import settings


class LocalEmbedder:
    """Real semantic embeddings (runs on your CPU, no API key). Downloads the model once."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class HashEmbedder:
    """Offline TEST mode: bag-of-words hashed into a fixed vector. Matches words, not meaning.
    Only for checking that the pipeline works without downloading a model."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in re.findall(r"\w+", text.lower()):
                vec[zlib.crc32(tok.encode()) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def get_embedder():
    if settings.embed_backend == "hash":
        return HashEmbedder()
    return LocalEmbedder(settings.embed_model)

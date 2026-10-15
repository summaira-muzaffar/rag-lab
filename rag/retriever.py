"""STEP 5 - RETRIEVE: embed the question and fetch the most similar chunks."""


class Retriever:
    def __init__(self, embedder, store, top_k: int):
        self.embedder, self.store, self.top_k = embedder, store, top_k

    def retrieve(self, question: str, k: int | None = None):
        q_vec = self.embedder.embed([question])[0]
        return self.store.search(q_vec, k or self.top_k)

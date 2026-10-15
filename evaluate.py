"""Measure retrieval quality:  python evaluate.py
hit@k  = the right source file appeared in the top-k results
answer = a key fact from the right passage appeared in the retrieved text"""
import json

from rag.pipeline import RAG

if __name__ == "__main__":
    rag = RAG(with_llm=False)
    cases = json.load(open("eval/questions.json", encoding="utf-8"))
    src_hits = kw_hits = 0
    for c in cases:
        hits = rag.retriever.retrieve(c["question"])
        ok_src = any(h["source"] == c["expected_source"] for h in hits)
        ok_kw = any(c["keyword"].lower() in h["text"].lower() for h in hits)
        src_hits += ok_src
        kw_hits += ok_kw
        print(f"{'PASS' if ok_src and ok_kw else 'FAIL'}  {c['question']}")
    n = len(cases)
    print(f"\nsource hit@{rag.retriever.top_k}: {src_hits}/{n}   fact retrieved: {kw_hits}/{n}")

"""Ask questions in the terminal.
   python ask.py "How many sick days do I get?"     (one question)
   python ask.py                                    (interactive)"""
import sys

from rag.pipeline import RAG


def show(rag: RAG, q: str):
    ans = rag.ask(q)
    print(f"\nA: {ans.text}\n")
    for i, h in enumerate(ans.sources, start=1):
        print(f"  [{i}] {h['source']}  (similarity {h['score']})")


if __name__ == "__main__":
    rag = RAG()
    if len(sys.argv) > 1:
        show(rag, " ".join(sys.argv[1:]))
    else:
        while (q := input("\nQuestion (empty to quit): ").strip()):
            show(rag, q)

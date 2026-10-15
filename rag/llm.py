"""STEP 6 - GENERATE: give the LLM the question + retrieved chunks and ask it to answer from them."""
import os

from config import settings

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite the passages you used like [1] or [2]. "
    "If the context does not contain the answer, reply exactly: I don't know based on the documents."
)


def build_user_message(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


class AnthropicLLM:
    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    def answer(self, question: str, hits: list[dict]) -> str:
        msg = self.client.messages.create(
            model=settings.llm_model,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_message(question, hits)}],
        )
        return msg.content[0].text


class GeminiLLM:
    def __init__(self):
        from dotenv import load_dotenv
        from google import genai

        load_dotenv()
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    def answer(self, question: str, hits: list[dict]) -> str:
        from google.genai import types

        reply = self.client.models.generate_content(
            model=self.model,
            contents=build_user_message(question, hits),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=600,
            ),
        )
        return reply.text or "(no answer returned)"


class EchoLLM:
    """No-LLM mode: just shows what was retrieved. Useful for debugging retrieval."""

    def answer(self, question: str, hits: list[dict]) -> str:
        top = hits[0]["text"] if hits else "(nothing retrieved)"
        return f"[echo mode, no LLM] Top passage:\n{top}"


def get_llm():
    if settings.llm_backend == "echo":
        return EchoLLM()
    from dotenv import load_dotenv

    load_dotenv()
    if os.getenv("GEMINI_API_KEY"):
        return GeminiLLM()
    return AnthropicLLM()

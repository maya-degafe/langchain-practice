"""Terminal chat workflow for the educational PDF RAG example."""

from __future__ import annotations

import argparse

from openai import OpenAI

from app.config import ConfigError, Settings, get_settings
from app.db import DatabaseError
from app.retrieve import print_matches, retrieve_chunks


class SimpleLLM:
    """A tiny answer-generation layer that can be swapped later."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def answer(self, question: str, matches: list[dict]) -> str:
        """Generate an answer from the retrieved chunks."""

        if not matches:
            return "I could not find any matching chunks. Try ingesting PDFs first or ask a broader question."

        if self.settings.llm_backend == "openai":
            return self._answer_with_openai(question, matches)

        return self._answer_with_context_only(matches)

    def _answer_with_openai(self, question: str, matches: list[dict]) -> str:
        """Use a small OpenAI chat completion call when configured."""

        if not self.settings.openai_api_key:
            raise ConfigError(
                "OPENAI_API_KEY is required when LLM_BACKEND=openai. "
                "Set it in .env or switch LLM_BACKEND to context_only."
            )

        client = OpenAI(api_key=self.settings.openai_api_key)
        completion = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question using only the provided PDF context. "
                        "If the answer is not in the context, say so clearly."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_prompt(question, matches),
                },
            ],
        )
        return completion.choices[0].message.content or "No answer was returned by the LLM."

    def _answer_with_context_only(self, matches: list[dict]) -> str:
        """Return a simple context-based answer with no external API calls."""

        summaries = []
        for item in matches[:3]:
            snippet = " ".join(item["content"].split())
            summaries.append(f"- {item['source_file']} (chunk {item['chunk_index']}): {snippet[:280]}")
        return (
            "LLM_BACKEND=context_only, so this is a lightweight answer based on the top retrieved chunks:\n"
            + "\n".join(summaries)
        )

    @staticmethod
    def _build_prompt(question: str, matches: list[dict]) -> str:
        """Build a compact prompt that keeps the LLM layer easy to replace later."""

        context_blocks = []
        for item in matches:
            context_blocks.append(
                f"Source: {item['source_file']} | Chunk: {item['chunk_index']}\n{item['content']}"
            )
        context = "\n\n".join(context_blocks)
        return f"Question: {question}\n\nContext:\n{context}"


def main() -> None:
    """CLI entry point for asking questions in the terminal."""

    parser = argparse.ArgumentParser(description="Ask a question against the ingested PDF vector store.")
    parser.add_argument("--question", help="Optional question. If omitted, the app asks interactively.")
    args = parser.parse_args()

    question = args.question or input("Question: ").strip()
    if not question:
        print("Please provide a question.")
        return

    try:
        settings = get_settings(require_llm=False)
        matches = retrieve_chunks(question)
        print_matches(matches)
        llm = SimpleLLM(settings)
        print("Answer:\n")
        print(llm.answer(question, matches))
    except (ConfigError, DatabaseError) as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()

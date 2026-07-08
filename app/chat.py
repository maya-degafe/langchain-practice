"""Terminal chat workflow for the educational PDF RAG example."""

from __future__ import annotations

import argparse

from app.config import ConfigError, Settings, get_settings
from app.db import DatabaseError
from app.retrieve import retrieve_chunks


class CapyLLM:
    # LLM setup
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.system_prompt = (
            "You are a helpful, enthusiastic assistant that ONLY answers questions using the provided PDF context."
            "If the answer is not in the context, say:"
            "Sorry, I do not know the answer to that question. Try rewording the question, or reaching out to our Help Desk at 888-888-8888."
        )
        self.min_similarity = 0.22

    # Main answer pipeline: filter by similarity/question size, then call the LLM to answer
    def answer(self, question: str, matches: list[dict]) -> str:
        """Generate an answer from the retrieved chunks."""

        q = question.strip()
        if not q: # if question is empty
            # ideally a user shouldnt be able to get here, but j in case
            return "Please provide a question for me to answer."
        if len(q) > 700:
            return "Your question is too long. Please shorten it to 700 characters or less."        

        strong = [m for m in matches if float(m.get("similarity", 0.0)) >= self.min_similarity]
        if not strong:
            return "Sorry, I do not know the answer to that question. Try rewording the question, or reaching out to our Help Desk at 888-888-8888."
        return self._answer_with_ollama(q, strong[:5])
    
    # Ollama generation
    def _answer_with_ollama(self, question: str, matches: list[dict]) -> str:

        try:
            from langchain_ollama import ChatOllama
        except Exception as exc:
            raise ConfigError(
                "langchain-ollama is required for LLM_BACKEND=ollama. Install with: uv add langchain-ollama"
            ) from exc

        model = getattr(self.settings, "gemma_model", "gemma3:4b")
        base_url = getattr(self.settings, "gemma_base_url", "http://127.0.0.1:11434")

        llm = ChatOllama(model=model, base_url=base_url, temperature=0.2)
        prompt = self._build_prompt(question, matches)
        response = llm.invoke(
            [
                ("system", self.system_prompt),
                ("human", prompt),
            ]
        )
        text = response.content if isinstance(response.content, str) else str(response.content)
        # enforce fallback message
        if "not in the context" in text.lower() or "do not provide" in text.lower():
            return "Sorry, I do not know the answer to that question. Try rewording the question, or reaching out to our Help Desk at 888-888-8888."
        return text

    @staticmethod
    def _build_prompt(question: str, matches: list[dict]) -> str:
        """Build a compact prompt that keeps the LLM layer easy to replace later."""

        context_blocks = []
        for item in matches[:5]: # limit to top 5 matches
            context_blocks.append(
                f"Source: {item['source_file']} | Chunk: {item['chunk_index']}\n{item['content']}"
            )
        context = "\n\n".join(context_blocks)
        return f"Question: {question}\n\nContext:\n{context}"

# cli entrypoint
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
        llm = CapyLLM(settings)
        print("Answer:\n")
        print(llm.answer(question, matches))
    except (ConfigError, DatabaseError) as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
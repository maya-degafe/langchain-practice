"""Terminal chat workflow for the educational PDF RAG example."""

from __future__ import annotations

import argparse
from typing import Dict

from app.config import ConfigError, Settings, get_settings
from app.db import DatabaseError
from app.retrieve import retrieve_chunks

# imports for temp history
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from pathlib import Path

def _load_system_prompt() -> str:
    prompt_path = Path(__file__).parent / "prompts" / "system_prompt.txt"
    return prompt_path.read_text(encoding="utf-8").strip()


class CapyLLM:
    # LLM setup
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.min_similarity = 0.22
        self.fallback = (
            "Sorry, I do not know the answer to that question. Try rewording the question, " 
            "or reaching out to our Help Desk at 888-888-8888."
        )

        try:
            from langchain_ollama import ChatOllama
        except Exception as exc:
            raise ConfigError(
                "langchain-ollama is required for LLM_BACKEND=ollama. Install with: uv add langchain-ollama"
            ) from exc
        
        model = getattr(self.settings, "ollama_model", "gemma3:4b")
        base_url = getattr(self.settings, "ollama_base_url", "http://127.0.0.1:11434")
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=0.2)

        self.system_prompt = _load_system_prompt()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="history"),
                (
                    "human",
                    "Question: {question}\n\nContext:\n{context}",
                ),
            ]
        )
        self.chain = self.prompt | self.llm

        # in memory session store
        self._store: Dict[str, InMemoryChatMessageHistory] = {}   

        def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self._store:
                self._store[session_id] = InMemoryChatMessageHistory()
            return self._store[session_id]
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            get_by_session_id,
            input_messages_key="question",
            history_messages_key="history",
            )    

    # Main answer pipeline: filter by similarity/question size, then call the LLM to answer
    def answer(self, question: str, session_id: str = "cli") -> str:
        q = question.strip()
        if not q:
            return "Please provide a question for me to answer."
        if len(q) > 700:
            return "Your question is too long. Please shorten it to 700 characters or less."

        matches = retrieve_chunks(q)
        strong = [m for m in matches if float(m.get("similarity", 0.0)) >= self.min_similarity]

        # Do not hard-fail immediately; allow conversational replies when retrieval is weak
        context = self._build_context(strong[:5]) if strong else "No relevant PDF context retrieved."

        response = self.chain_with_history.invoke(
            {"question": q, "context": context},
            config={"configurable": {"session_id": session_id}},
        )
        text = response.content if isinstance(response.content, str) else str(response.content)

        # Keep strict fallback only when there is no retrieval support and model declines
        if not strong and any(x in text.lower() for x in ["not in the context", "don't know", "do not know"]):
            return self.fallback
        return text


    @staticmethod
    def _build_context(matches: list[dict]) -> str:
        blocks = []
        for item in matches:
            blocks.append(
                f"Source: {item['source_file']} | Chunk: {item['chunk_index']}\n{item['content']}"
            )
        return "\n\n".join(blocks)
    
# cli entrypoint
def main() -> None:
    """CLI entry point for asking questions in the terminal."""

    parser = argparse.ArgumentParser(description="Ask a question against the ingested PDF vector store.")
    parser.add_argument("--question", help="Optional question. If omitted, the app asks interactively.")
    parser.add_argument("--session-id", default="cli", help="Session id for chat memory.")
    args = parser.parse_args()

    try:
        settings = get_settings(require_llm=False)
        llm = CapyLLM(settings)

        # One-shot mode
        if args.question:
            print("Answer:\n")
            print(llm.answer(args.question, session_id=args.session_id))
            return

        # Interactive mode (single prompt loop)
        while True:
            q = input("Question: ").strip()
            if not q:
                print("Please provide a question.")
                continue
            if q.lower() in {"exit", "quit"}:
                break

            print("\nAnswer:\n")
            print(llm.answer(q, session_id=args.session_id))
            print()

    except (ConfigError, DatabaseError) as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
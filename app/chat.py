from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from sqlalchemy import text

from app.config import ConfigError, Settings, get_settings
from app.db import DatabaseError
from app.intent import classify_intent
from app.retrieve import retrieve_chunks

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


def _load_system_prompt() -> str:
    prompt_path = Path(__file__).parent / "prompts" / "system_prompt.txt"
    return prompt_path.read_text(encoding="utf-8").strip()


import re


def _strip_markdown(text: str) -> str:
    # removes headers (#, ##, ###...)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    # removes **bold** and __bold__
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    # removes *italic* and _italic_
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    # normalizes bullet markers (* or -) to a simple dash
    text = re.sub(r"^(\s*)[\*\-]\s+", r"\1- ", text, flags=re.MULTILINE)
    return text
class CapyLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        # retrieval thresholds
        self.min_similarity = 0.22
        self.min_top_similarity = 0.30

        self.domain_redirect = (
            "I can help with questions based on the provided documents. "
            "What would you like to know?"
        )

        self.fallback = (
            "Sorry, I do not know the answer to that question. "
            "Try rewording the question, or reaching out to our Help Desk at 888-888-8888."
        )

        self.no_match = (
            "I couldn't find relevant information in the provided documents for that question. "
            "Please try rewording it or asking about a more specific topic."
        )

        self.repair_redirect = (
            "You're right — let me reset. I can help with information from our documents. "
            "What would you like help with?"
        )

        self.safety_redirect = (
            "If this is an urgent emergency, please contact local emergency services right away. "
            "I can help with information from our documents."
        )

        self.escalation = (
            "I'll connect you with a CapAir customer service representative. "
            "Please call our Help Desk at 888-888-8888, and have your booking "
            "reference ready. Is there anything else I can help you find in the meantime?"
        )

        try:
            from langchain_anthropic import ChatAnthropic
        except Exception as exc:
            raise ConfigError(
                "langchain-anthropic is required. "
                "Install with: uv add langchain-anthropic"
            ) from exc

        if not self.settings.anthropic_api_key:
            raise ConfigError(
                "ANTHROPIC_API_KEY is required. Set it in your .env file."
            )

        self.llm = ChatAnthropic(
            model=self.settings.anthropic_model,
            base_url=self.settings.anthropic_base_url,
            api_key=self.settings.anthropic_api_key,
            temperature=0.2,
            max_tokens=300,
        )

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

    def _is_escalation_request(self, q: str) -> bool:
        """Detect explicit requests to reach a human representative."""
        lowered = q.lower().strip()
        if lowered in {"agent", "human", "representative", "rep"}:
            return True
        phrases = [
            "talk to a person",
            "speak to a human",
            "speak to an agent",
            "talk to an agent",
            "real person",
            "customer service rep",
            "talk to a human",
            "talk to someone",
        ]
        return any(phrase in lowered for phrase in phrases)

    def _is_competitor_request(self, q: str) -> bool:
        """Detect requests to compare or recommend non-CapAir airlines."""
        lowered = q.lower().strip()
        if "airline" not in lowered and "airlines" not in lowered:
            return False
        phrases = [
            "recommend a different airline",
            "recommend another airline",
            "recommend other airlines",
            "different airline",
            "another airline",
            "other airlines",
            "compare capair",
        ]
        return any(phrase in lowered for phrase in phrases)

    def answer(self, question: str, session_id: str = "cli") -> str:
        """Route the user message before deciding whether to retrieve and call the LLM."""
        q = question.strip()

        if not q:
            return "Please provide a question for me to answer."

        if len(q) > 700:
            return "Your question is too long. Please shorten it to 700 characters or less."

        # --- Escalation: explicit request for a human takes priority ---
        if self._is_escalation_request(q):
            return self.escalation

        if self._is_competitor_request(q):
            return (
                "I can't recommend, compare, or endorse other airlines. "
                "I can help with CapAir policies, or connect you with a CapAir "
                "customer service representative at 888-888-8888."
            )

        intent = classify_intent(q)

        # Safety and repair intents always take priority.
        if intent.intent == "safety":
            return self.safety_redirect

        if intent.intent == "repair":
            return self.repair_redirect

        # Retrieve first so a strong match can override a weak/incorrect intent label.
        matches = retrieve_chunks(q)
        has_relevant = self._has_relevant_match(matches)

        # Only redirect as "other" if retrieval ALSO fails to find anything relevant.
        # This rescues short/vague-but-on-topic questions (e.g. "tell me about CO2")
        # that the intent classifier mislabels as "other".
        if intent.intent == "other" and not has_relevant:
            return self.domain_redirect

        if not has_relevant:
            return self.no_match

        strong = [m for m in matches if float(m.get("similarity", 0.0)) >= self.min_similarity]

        if not strong:
            return self.no_match

        context = self._build_context(strong[:5])

        try:
            response = self.chain_with_history.invoke(
                {"question": q, "context": context},
                config={"configurable": {"session_id": session_id}},
            )
        except Exception as exc:
            backend_error = self._friendly_llm_error(exc)
            if backend_error:
                return backend_error
            raise

        text = response.content if isinstance(response.content, str) else str(response.content)

        if any(x in text.lower() for x in ["not in the context", "don't know", "do not know"]):
            return self.fallback
        
        return _strip_markdown(text)

    

    def _has_relevant_match(self, matches: list[dict]) -> bool:
        """Check whether the top retrieval hit is strong enough to trust."""
        if not matches:
            return False
        top_score = float(matches[0].get("similarity", 0.0))
        return top_score >= self.min_top_similarity

    @staticmethod
    def _build_context(matches: list[dict]) -> str:
        """Format retrieved chunks into LLM context."""
        blocks = []
        for item in matches:
            blocks.append(
                f"Source: {item['source_file']} | Chunk: {item['chunk_index']}\n{item['content']}"
            )
        return "\n\n".join(blocks)

    def _friendly_llm_error(self, exc: Exception) -> str | None:
        """Return a user-friendly backend error message when known failures occur."""
        message = str(exc)
        lowered = message.lower()
        backend = (self.settings.llm_backend or "unknown").strip().lower()
        exc_name = type(exc).__name__.lower()

        if "connection refused" in lowered or "connecterror" in exc_name:
            return (
                f"Could not reach the {backend} backend. "
                "Make sure the service is running and try again."
            )

        auth_markers = [
            "authenticationerror",
            "invalid_api_key",
            "permission_denied_error",
            "signature expired",
            "unauthorized",
            "error code: 401",
        ]
        if any(marker in lowered for marker in auth_markers):
            return (
                "The model backend rejected authentication (401). "
                "Your API key or temporary token may be expired. "
                "Refresh your credentials in .env and retry."
            )

        return None


def main() -> None:
    """CLI entry point for asking questions in the terminal."""
    parser = argparse.ArgumentParser(description="Ask a question against the ingested PDF vector store.")
    parser.add_argument("--question", help="Optional question. If omitted, the app asks interactively.")
    parser.add_argument("--session-id", default="cli", help="Session id for chat memory.")
    args = parser.parse_args()

    try:
        settings = get_settings(require_llm=False)
        llm = CapyLLM(settings)

        if args.question:
            print("Answer:\n")
            print(llm.answer(args.question, session_id=args.session_id))
            return

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
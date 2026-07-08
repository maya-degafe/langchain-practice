"""Embedding and retrieval workflow for the PDF RAG example."""

from __future__ import annotations

import argparse
from typing import Iterable

from sentence_transformers import SentenceTransformer

from sklearn.preprocessing import normalize

from app.config import ConfigError, get_settings
from app.db import DatabaseError, get_connection, initialize_database, search_similar_chunks


class EmbeddingService:
    """Service for embedding text and queries into dense vectors."""

    def __init__(self, model_name: str, *, dimensions: int | None = None) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed texts as normalized dense vectors for pgvector cosine search."""

        text_list = list(texts)
        if not text_list:
            return []
        vectors = self.model.encode(text_list, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:

        return self.embed_texts([text])[0]


def retrieve_chunks(query: str, *, top_k: int | None = None) -> list[dict]:
    """Run vector similarity search for a natural-language query."""
    # load settings
    settings = get_settings()
    # embed user query into the same vector space as document chunks
    embedding_service = EmbeddingService(settings.embedding_model, dimensions=settings.embedding_dimension)
    query_embedding = embedding_service.embed_query(query)

    # open db connection, unsure schema exists, run similarity search
    with get_connection(settings) as connection:
        initialize_database(connection, settings)
        return search_similar_chunks(connection, query_embedding, top_k=top_k or settings.top_k)

# output formatting
def print_matches(matches: list[dict]) -> None:
    """Print retrieved chunks in a beginner-friendly format."""

    if not matches:
        print("No matching chunks were found. Ingest PDFs first with `python -m app.ingest`.")
        return

    print("\nRetrieved chunks:\n")
    for item in matches:
        similarity = float(item["similarity"])
        print(f"- source: {item['source_file']} | chunk: {item['chunk_index']} | similarity: {similarity:.4f}")
        print(item["content"].strip())
        print()


def main() -> None:
    """CLI entry point for testing retrieval without answer generation."""

    parser = argparse.ArgumentParser(description="Run vector similarity search against stored PDF chunks.")
    parser.add_argument("--query", required=True, help="Question or search query to run against the vector store.")
    parser.add_argument("--top-k", type=int, default=None, help="Optional number of chunks to retrieve.")
    args = parser.parse_args()

    try:
        matches = retrieve_chunks(args.query, top_k=args.top_k)
        print_matches(matches)
    except (ConfigError, DatabaseError) as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()

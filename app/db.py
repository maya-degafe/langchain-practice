"""Database helpers for PostgreSQL + pgvector."""

from __future__ import annotations

from typing import Iterable

import psycopg
from pgvector.psycopg import register_vector
from pgvector.psycopg.vector import Vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.config import Settings


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    source_file TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_file, chunk_index)
);

CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
ON document_chunks USING hnsw (embedding vector_cosine_ops);
"""


class DatabaseError(RuntimeError):
    """Raised when the app cannot use the PostgreSQL database."""


def get_connection(settings: Settings) -> psycopg.Connection:
    """Open a PostgreSQL connection."""

    try:
        return psycopg.connect(settings.database_url, autocommit=True, row_factory=dict_row)
    except psycopg.OperationalError as exc:
        raise DatabaseError(
            "Could not connect to PostgreSQL. Make sure Docker Compose is running "
            "and your .env values match docker-compose.yml."
        ) from exc


def initialize_database(connection: psycopg.Connection, settings: Settings) -> None:
    """Enable pgvector and create the example table if needed."""

    schema_sql = SCHEMA_SQL.replace("VECTOR(384)", f"VECTOR({settings.embedding_dimension})")
    with connection.cursor() as cursor:
        cursor.execute(schema_sql)
    register_vector(connection)


def replace_document_chunks(
    connection: psycopg.Connection,
    source_file: str,
    rows: Iterable[tuple[int, str, dict, list[float]]],
) -> int:
    """Replace all stored chunks for one source file."""

    row_list = list(rows)
    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM document_chunks WHERE source_file = %s", (source_file,))
        if not row_list:
            return 0
        cursor.executemany(
            """
            INSERT INTO document_chunks (source_file, chunk_index, content, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (source_file, chunk_index, content, Jsonb(metadata), embedding)
                for chunk_index, content, metadata, embedding in row_list
            ],
        )
    return len(row_list)


def search_similar_chunks(
    connection: psycopg.Connection,
    query_embedding: list[float],
    *,
    top_k: int,
) -> list[dict]:
    """Return the top-k most similar chunks using pgvector cosine distance."""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                source_file,
                chunk_index,
                content,
                metadata,
                1 - (embedding <=> %s) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (Vector(query_embedding), Vector(query_embedding), top_k),
        )
        return list(cursor.fetchall())

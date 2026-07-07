"""PDF ingestion pipeline for the educational RAG example."""

from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import ConfigError, get_settings
from app.db import DatabaseError, get_connection, initialize_database, replace_document_chunks
from app.retrieve import EmbeddingService


class IngestError(RuntimeError):
    """Raised when PDF ingestion cannot continue."""



def find_pdf_files(data_dir: Path) -> list[Path]:
    """Return all PDFs from the configured local data directory."""

    if not data_dir.exists():
        raise IngestError(f"Data directory does not exist: {data_dir}")

    pdf_files = sorted(path for path in data_dir.iterdir() if path.suffix.lower() == ".pdf")
    if not pdf_files:
        raise IngestError(f"No PDF files were found in {data_dir}. Add at least one PDF and try again.")
    return pdf_files



def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from one PDF file using pypdf."""

    reader = PdfReader(str(pdf_path))
    page_text = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(part.strip() for part in page_text if part.strip())
    if not text:
        raise IngestError(f"No text could be extracted from {pdf_path.name}.")
    return text



def split_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split extracted text into overlapping chunks for retrieval."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)



def ingest_pdfs() -> None:
    """Read PDFs, embed their chunks, and store them in PostgreSQL."""

    settings = get_settings()
    pdf_files = find_pdf_files(settings.data_dir)
    embedding_service = EmbeddingService(settings.embedding_model, dimensions=settings.embedding_dimension)

    with get_connection(settings) as connection:
        initialize_database(connection, settings)

        total_chunks = 0
        for pdf_path in pdf_files:
            text = extract_pdf_text(pdf_path)
            chunks = split_text(
                text,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            embeddings = embedding_service.embed_texts(chunks)
            rows = [
                (
                    index,
                    chunk,
                    {
                        "source_path": str(pdf_path),
                        "file_name": pdf_path.name,
                    },
                    embedding,
                )
                for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
            ]
            inserted = replace_document_chunks(connection, pdf_path.name, rows)
            total_chunks += inserted
            print(f"Ingested {inserted} chunks from {pdf_path.name}")

    print(f"Finished ingesting {len(pdf_files)} PDF(s) and {total_chunks} total chunk(s).")



def main() -> None:
    """CLI entry point for the PDF ingestion flow."""

    try:
        ingest_pdfs()
    except (ConfigError, DatabaseError, IngestError) as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()

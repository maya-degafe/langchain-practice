# langchain-practice

A small, beginner-friendly Python project for learning a local PDF RAG (Retrieval-Augmented Generation) workflow with:

- **LangChain** for text splitting
- **PostgreSQL + pgvector** for vector storage and similarity search
- **pypdf** for PDF parsing
- **a simple local hashing embedder** for embeddings
- a very small, swappable **LLM layer** for answer generation

The goal is to keep the code easy to read and easy to modify.

## What this project does

1. Reads PDF files from the local `data/` directory
2. Extracts text from each PDF
3. Splits the text into chunks
4. Generates embeddings for those chunks
5. Stores chunks, metadata, and embeddings in PostgreSQL with pgvector
6. Runs vector similarity search for a user question
7. Prints the matched chunks so you can see what was retrieved
8. Generates an answer with a very simple LLM wrapper

## Project structure

```text
.
├── app/
│   ├── __init__.py
│   ├── chat.py
│   ├── config.py
│   ├── db.py
│   ├── ingest.py
│   ├── intent.py
│   ├── normalize.py
│   ├── retrieve.py
│   └── safety_rules.py
├── data/
│   └── growing_veggies_in_fl.pdf
├── .env.example
├── docker-compose.yml
├── Makefile
├── README.md
└── requirements.txt
```

## Prerequisites

- Python 3.11+
- Docker + Docker Compose

## Setup

### 1) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3) Create your local environment file

```bash
cp .env.example .env
```

The default `.env.example` values are already set up for the local Docker database.

### 4) Start PostgreSQL with pgvector

```bash
docker compose up -d
```

Or use:

```bash
make up
```

### 5) Add PDFs

Put one or more PDF files into the local `data/` directory.

Example:

```bash
cp /path/to/your/file.pdf data/
```

## Usage

### Ingest PDFs into PostgreSQL

```bash
python -m app.ingest
```

This command will:

- connect to PostgreSQL
- enable the `vector` extension if needed
- create the table if it does not exist
- read all PDFs in `data/`
- split them into chunks
- generate embeddings
- store chunks + metadata + embeddings in the database

Or use:

```bash
make ingest
```

### Run retrieval only

```bash
python -m app.retrieve --query "What is this document about?"
```

This prints the top matches, including source file names and similarity scores.

### Ask a question in the terminal

```bash
python -m app.chat --question "What are the main ideas in the PDFs?"
```

Or use interactive mode:

```bash
python -m app.chat session-id
```

Or use:

```bash
make chat QUESTION="What are the main ideas in the PDFs?"
```

## Embeddings

This starter uses `EMBEDDING_MODEL=simple-hashing` by default. That keeps the project fully local and avoids any model downloads, which is helpful for a first RAG practice project.

The embedding code is isolated in `app/retrieve.py`, so you can later swap it for a sentence-transformer, OpenAI embeddings, or another provider.

## Simple LLM setup

This project keeps the answer-generation layer intentionally small.

### Default mode: `context_only`

By default, the app uses:

```env
LLM_BACKEND=context_only
```

## Retrieval flow test

A simple learning flow:

1. start the database
2. add one PDF to `data/`
3. run `python -m app.ingest`
4. run `python -m app.retrieve --query "your question"`
5. inspect the printed chunks and similarity scores
6. run `python -m app.chat --question "your question"`

This makes it easy to see what was matched before looking at the final answer.

## Database schema

The app creates a single table named `document_chunks`.

Important columns:

- `source_file`: which PDF the chunk came from
- `chunk_index`: the order of the chunk inside the PDF
- `content`: the chunk text
- `metadata`: JSON metadata such as page count and source path
- `embedding`: pgvector embedding used for similarity search

The app also enables the `vector` extension automatically.

## Notes

- `.env` is ignored by Git so secrets stay out of the repository.
- The default embedding backend is fully local and does not need any model download.
- If you swap to another embedding implementation later, keep `EMBEDDING_DIMENSION` in sync with the vector size you store in pgvector.

## Stopping the database

```bash
docker compose down
```

Or:

```bash
make down
```

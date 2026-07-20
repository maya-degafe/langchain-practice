"""Application configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" # where the data is read from
ENV_PATH = REPO_ROOT / ".env"

load_dotenv(ENV_PATH, override=False)


class ConfigError(ValueError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Small container for environment-based application settings."""

    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: int
    data_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_model: str
    embedding_dimension: int
    llm_backend: str
    # --- LLM backend settings (new) ---
    ollama_model: str
    ollama_base_url: str
    bedrock_model: str
    bedrock_base_url: str
    bedrock_api_key: str
    # --- Anthropic (Claude via Bedrock Messages API) ---
    anthropic_model: str
    anthropic_base_url: str
    anthropic_api_key: str

    @property
    def database_url(self) -> str:
        """Build a PostgreSQL connection string for psycopg."""

        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


def _get_required_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""

    value = os.getenv(name)
    if value:
        return value
    raise ConfigError(
        f"Missing required environment variable: {name}. "
        "Copy .env.example to .env and update the values if needed."
    )


def get_settings(*, require_llm: bool = False) -> Settings:
    """Load application settings from environment variables.

    Parameters
    ----------
    require_llm:
        When True, validates LLM-specific settings too.
    """

    data_dir_value = os.getenv("DATA_DIR")
    data_dir = (REPO_ROOT / data_dir_value).resolve() if data_dir_value else DEFAULT_DATA_DIR

    settings = Settings(
        postgres_user=_get_required_env("POSTGRES_USER"),
        postgres_password=_get_required_env("POSTGRES_PASSWORD"),
        postgres_db=_get_required_env("POSTGRES_DB"),
        postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
        postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
        data_dir=data_dir,
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")), # chars per chunk
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")), # shared text between chunks
        top_k=int(os.getenv("TOP_K", "5")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "simple-hashing").strip(),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
        llm_backend=os.getenv("LLM_BACKEND", "context_only").strip().lower(),
        # --- LLM backend settings (new) ---
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        bedrock_model=os.getenv("BEDROCK_MODEL", "openai.gpt-oss-120b"),
        bedrock_base_url=os.getenv(
            "BEDROCK_BASE_URL",
            os.getenv("OPENAI_BASE_URL", "https://bedrock-mantle.us-east-1.api.aws/v1"),
        ),
        bedrock_api_key=os.getenv("BEDROCK_API_KEY")
        or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        or os.getenv("OPENAI_API_KEY")
        or "",
        # --- Anthropic (Claude via Bedrock Messages API) ---
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        anthropic_base_url=os.getenv(
            "ANTHROPIC_BASE_URL",
            "https://bedrock-mantle.us-east-1.api.aws/anthropic",
        ),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    )

    # enforce that chunk overlap is smaller than chunk size
    if settings.chunk_overlap >= settings.chunk_size:
        raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    return settings
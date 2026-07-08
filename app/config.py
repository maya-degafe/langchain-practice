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
        embedding_model=os.getenv("EMBEDDING_MODEL", "simple-hashing").strip().lower(),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
        llm_backend=os.getenv("LLM_BACKEND", "context_only").strip().lower(),
    )

    # enforce that chunk overlap is smaller than chunk size
    if settings.chunk_overlap >= settings.chunk_size:
        raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    return settings

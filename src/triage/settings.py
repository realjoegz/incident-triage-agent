"""Settings, read from `TRIAGE_*` environment variables."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from triage.models import SearchMode

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIAGE_")

    # The default is the database of compose.yaml, reached from the host.
    database_url: str = "postgresql://triage:triage@127.0.0.1:54329/triage"
    data_dir: Path = Path("data")
    eval_dir: Path = Path("eval")

    # Where fastembed keeps the model file. Unset means its own default under the temp dir.
    embedding_cache_dir: Path | None = None

    model: str = "claude-haiku-5-5"
    effort: Literal["low", "medium", "high"] = "low"
    search_mode: SearchMode = "hybrid"
    top_k: int = 5

    # `replay` answers from the recorded responses of the evaluation and never calls the API.
    llm: Literal["anthropic", "replay"] = "anthropic"

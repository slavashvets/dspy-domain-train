from pathlib import Path

from pydantic import AnyHttpUrl, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureOpenAIModelSettings(BaseModel):
    """Complete configuration for a single Azure OpenAI LLM."""

    endpoint: AnyHttpUrl
    api_key: str
    api_version: str
    deployment: str
    model_type: str = "chat"
    temperature: float = 0.0
    max_tokens: int | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="DSPY_",
        env_nested_delimiter="__",
    )

    eval: AzureOpenAIModelSettings
    refine: AzureOpenAIModelSettings

    devset_path: Path = Path("data/dev.json")
    prompt_path: Path = Path("prompts/p0.txt")

    max_iters: int = 6
    tol: float = 5e-3
    refiner_candidates: int = 5
    refiner_retries: int = 1
    instr_max_len: int = 50


def get_settings() -> Settings:
    return Settings()

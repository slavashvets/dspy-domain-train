import os
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from .toml import ConfigPath, settings_toml_files

_PROFILE = os.getenv("DSPY_PROFILE", "local")


class AzureOpenAIModelSettings(BaseModel):
    """Complete configuration for a single Azure OpenAI LLM."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    endpoint: AnyHttpUrl
    api_key: SecretStr = Field(repr=False)
    api_version: str
    deployment: str
    model_type: Literal["chat", "responses", "text"] = "chat"
    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: int | None = Field(default=None, gt=0)


class Settings(BaseSettings):
    """Application settings loaded from base/profile TOML with env overrides."""

    model_config = SettingsConfigDict(
        extra="forbid",
        env_prefix="DSPY_",
        env_nested_delimiter="__",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(
                settings_cls,
                toml_file=settings_toml_files(_PROFILE),
                deep_merge=True,
            ),
            file_secret_settings,
        )

    eval: AzureOpenAIModelSettings
    refine: AzureOpenAIModelSettings

    train_path: ConfigPath
    dev_path: ConfigPath
    test_path: ConfigPath
    prompt_path: ConfigPath

    optimizer: Literal["copro", "gepa"] = "copro"
    num_threads: int = 4
    seed: int = 42
    max_train: int | None = None
    max_dev: int | None = None
    max_test: int | None = None

    copro_breadth: int = Field(default=10, ge=2)
    copro_depth: int = Field(default=3, ge=1)
    copro_init_temperature: float = Field(default=1.4, ge=0.0)

    gepa_auto: Literal["light", "medium", "heavy"] = "light"

    max_instruction_words: int = Field(default=150, ge=20)


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

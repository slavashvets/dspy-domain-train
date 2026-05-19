import os
from typing import Literal, Self

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from .toml import ConfigPath, settings_toml_files

_PROFILE = os.getenv("DSPY_PROFILE", "local")

type OptimizerBackend = Literal["copro", "simba", "gepa"]


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


class CoproSettings(BaseModel):
    """COPRO optimizer options."""

    model_config = ConfigDict(extra="forbid")

    breadth: int = Field(default=10, ge=2)
    depth: int = Field(default=3, ge=1)
    init_temperature: float = Field(default=1.0, ge=0.0)


class SimbaSettings(BaseModel):
    """SIMBA optimizer options."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=6, ge=1)
    bsize: int = Field(default=32, ge=4)
    num_candidates: int = Field(default=6, ge=2)


class GepaSettings(BaseModel):
    """GEPA optimizer options."""

    model_config = ConfigDict(extra="forbid")

    auto: Literal["light", "medium", "heavy"] = "light"


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

    optimizer: OptimizerBackend = "copro"
    num_threads: int = 4
    seed: int = 42
    max_train: int | None = None
    max_dev: int | None = None
    max_test: int | None = None
    max_instruction_words: int = Field(default=200, ge=20)

    copro: CoproSettings = CoproSettings()
    simba: SimbaSettings = SimbaSettings()
    gepa: GepaSettings = GepaSettings()

    @model_validator(mode="after")
    def _check_simba_bsize(self) -> Self:
        if self.optimizer != "simba" or not self.max_train:
            return self
        if self.max_train < self.simba.bsize:
            msg = (
                f"simba.bsize ({self.simba.bsize}) must be"
                f" <= max_train ({self.max_train})"
            )
            raise ValueError(msg)
        return self


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

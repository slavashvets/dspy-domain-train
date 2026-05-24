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

type OptimizerBackend = Literal["copro", "simba", "gepa", "srp"]


class AzureOpenAIModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    endpoint: AnyHttpUrl
    api_key: SecretStr = Field(repr=False)
    api_version: str
    deployment: str
    model_type: Literal["chat", "responses", "text"] = "chat"
    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: int | None = Field(default=None, gt=0)
    num_retries: int = Field(default=8, ge=0)


class CoproSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    breadth: int = Field(default=10, ge=2)
    depth: int = Field(default=3, ge=1)
    init_temperature: float = Field(default=1.0, ge=0.0)


class SimbaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=6, ge=1)
    bsize: int = Field(default=32, ge=4)
    num_candidates: int = Field(default=6, ge=2)


class GepaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto: Literal["light", "medium", "heavy"] = "light"


class SrpSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_iters: int = Field(default=6, ge=1)
    patience: int = Field(default=2, ge=1)
    max_examples: int | None = Field(default=None, ge=1)
    max_error_cases: int = Field(default=12, ge=1)
    num_candidates: int = Field(default=5, ge=1)
    candidate_retries: int = Field(default=1, ge=1)
    proposal_temperature: float = Field(default=1.0, ge=0.0)


class Settings(BaseSettings):
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
        profile = os.getenv("DSPY_PROFILE", "local")
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(
                settings_cls,
                toml_file=settings_toml_files(profile),
                deep_merge=True,
            ),
            file_secret_settings,
        )

    eval: AzureOpenAIModelSettings
    refine: AzureOpenAIModelSettings

    train_path: ConfigPath
    dev_path: ConfigPath
    test_path: ConfigPath
    initial_prompt_path: ConfigPath | None = None

    optimizer: OptimizerBackend = "srp"
    num_threads: int = 4
    seed: int = 42
    compare_baseline: bool = False
    max_train: int | None = None
    max_dev: int | None = None
    max_test: int | None = None

    copro: CoproSettings = Field(default_factory=CoproSettings)
    simba: SimbaSettings = Field(default_factory=SimbaSettings)
    gepa: GepaSettings = Field(default_factory=GepaSettings)
    srp: SrpSettings = Field(default_factory=SrpSettings)

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

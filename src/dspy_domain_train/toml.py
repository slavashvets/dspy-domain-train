from functools import cache
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator


def find_toml(name: str, start: Path | None = None) -> list[str]:
    """Walk up from start, or from CWD, looking for a TOML file."""
    root = start or Path.cwd()
    for directory in (root, *root.parents):
        candidate = directory / name
        if candidate.is_file():
            return [str(candidate)]
    return []


def settings_toml_files(profile: str) -> list[str]:
    """Return base settings, local secrets, and an optional profile overlay."""
    files = [*find_toml("settings.toml"), *find_toml("settings.local.toml")]
    if profile != "local":
        files.extend(find_toml(f"settings.{profile}.toml"))
    return files


@cache
def config_dir() -> Path:
    """Directory containing settings.toml, used for relative config paths."""
    paths = find_toml("settings.toml")
    return Path(paths[0]).parent if paths else Path.cwd()


def _resolve_config_path(value: Path) -> Path:
    if value.is_absolute():
        return value
    return (config_dir() / value).resolve()


ConfigPath = Annotated[Path, AfterValidator(_resolve_config_path)]

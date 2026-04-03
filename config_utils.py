"""Utilities for layered experiment configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PRESETS_DIR_NAME = "presets"
RUNTIME_DIR_NAME = "runtime"
GENERATED_DIR_NAME = "generated"

LATEST_LOCAL_CONFIG_RELATIVE = Path("generated/latest.local.yaml")
LATEST_LOCAL_META_RELATIVE = Path("generated/latest.local.meta.yaml")
LATEST_CLOUD_CONFIG_RELATIVE = Path("generated/latest.cloud.yaml")
LATEST_CLOUD_META_RELATIVE = Path("generated/latest.cloud.meta.yaml")
DEFAULT_PRESET_RELATIVE = Path("presets/balanced-mini-fires.yaml")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_yaml_map(path: Path) -> dict[str, Any]:
    """Read a YAML file and require a mapping payload."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config YAML non valida: {path}")
    return payload


def normalize_extends(value: Any) -> list[str]:
    """Normalize `extends` into a list of string paths."""
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError("La chiave 'extends' deve essere una stringa o una lista di stringhe")


def load_layered_config(config_path: Path, *, seen_paths: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML config and recursively merge inherited configs."""
    resolved_path = config_path.resolve()
    active_stack = seen_paths or set()
    if resolved_path in active_stack:
        raise ValueError(f"Ciclo rilevato nella configurazione: {resolved_path}")

    raw_config = read_yaml_map(resolved_path)
    extends_entries = normalize_extends(raw_config.pop("extends", None))

    merged: dict[str, Any] = {}
    next_stack = set(active_stack)
    next_stack.add(resolved_path)

    for entry in extends_entries:
        inherited_path = Path(entry)
        if not inherited_path.is_absolute():
            inherited_path = (resolved_path.parent / inherited_path).resolve()
        merged = deep_merge(merged, load_layered_config(inherited_path, seen_paths=next_stack))

    return deep_merge(merged, raw_config)


def _iter_config_paths(config_root: Path) -> list[Path]:
    if not config_root.exists():
        return []
    return sorted(
        path.relative_to(config_root)
        for path in config_root.rglob("*.yaml")
        if path.is_file()
    )


def is_metadata_config(relative_path: Path) -> bool:
    """Return True when a config file is metadata rather than runnable config."""
    return relative_path.name.endswith(".meta.yaml")


def is_runtime_config(relative_path: Path) -> bool:
    """Return True when a config file is a runtime-only override."""
    if relative_path.name.endswith(".runtime.yaml"):
        return True
    return bool(relative_path.parts) and relative_path.parts[0] == RUNTIME_DIR_NAME


def is_launchable_config(relative_path: Path) -> bool:
    """Return True when a config should be offered as a runnable experiment config."""
    return not is_metadata_config(relative_path) and not is_runtime_config(relative_path)


def list_launchable_configs(config_root: Path) -> list[str]:
    """List runnable configs relative to `config_root`."""
    return [path.as_posix() for path in _iter_config_paths(config_root) if is_launchable_config(path)]


def list_runtime_configs(config_root: Path) -> list[str]:
    """List runtime override configs relative to `config_root`."""
    return [path.as_posix() for path in _iter_config_paths(config_root) if is_runtime_config(path) and not is_metadata_config(path)]


def choose_default_launchable_config(config_root: Path) -> str | None:
    """Choose the preferred default local config for UI tools."""
    available_configs = list_launchable_configs(config_root)
    preferred_candidates = [LATEST_LOCAL_CONFIG_RELATIVE.as_posix(), DEFAULT_PRESET_RELATIVE.as_posix()]

    for candidate in preferred_candidates:
        if candidate in available_configs:
            return candidate
    return available_configs[0] if available_configs else None


def choose_default_cloud_launchable_config(config_root: Path) -> str | None:
    """Choose the required default cloud config for bundle generation."""
    available_configs = list_launchable_configs(config_root)
    preferred_candidates = [LATEST_CLOUD_CONFIG_RELATIVE.as_posix()]

    for candidate in preferred_candidates:
        if candidate in available_configs:
            return candidate
    return available_configs[0] if available_configs else None
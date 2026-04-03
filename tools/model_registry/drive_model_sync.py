"""Export/import YOLO model artifacts via Drive filesystem or OAuth Google Drive API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Include readonly scope so folder discovery is not limited to app-created files only.
DRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
]
DEFAULT_OAUTH_CREDENTIALS_FILE = "tools/model_registry/oauth_credentials.local.json"
ALL_RUN_LABEL_SELECTOR = "all"
ALL_RECURSIVE_RUN_LABEL_SELECTOR = "all-r"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML object in {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_with_persistent_root(path_value: str, persistent_root: Path) -> Path:
    raw = Path(path_value)
    if raw.is_absolute():
        return raw
    return (persistent_root / raw).resolve()


def resolve_local_artifact_reference(path_value: str, local_root: Path) -> Path:
    raw = Path(path_value.strip())
    if raw.is_absolute():
        return raw

    direct_candidate = (local_root / raw).resolve()
    if direct_candidate.exists():
        return direct_candidate

    matches = [candidate.resolve() for candidate in local_root.rglob(raw.name) if candidate.is_file()]
    if not matches:
        return direct_candidate

    if len(matches) == 1:
        return matches[0]

    local_matches = [candidate for candidate in matches if "local" in candidate.parts]
    if len(local_matches) == 1:
        return local_matches[0]

    export_matches = [candidate for candidate in matches if "exports" in candidate.parts]
    if len(export_matches) == 1:
        return export_matches[0]

    local_export_matches = [candidate for candidate in export_matches if "local" in candidate.parts]
    if len(local_export_matches) == 1:
        return local_export_matches[0]

    ranked_matches = sorted(matches, key=lambda candidate: (candidate.stat().st_mtime, str(candidate).lower()))
    newest_match = ranked_matches[-1]
    newest_mtime = newest_match.stat().st_mtime
    newest_ties = [candidate for candidate in ranked_matches if candidate.stat().st_mtime == newest_mtime]
    if len(newest_ties) == 1:
        return newest_match

    options = "\n".join(f"- {candidate}" for candidate in matches[:12])
    raise ValueError(
        f"Ambiguous artifact reference '{path_value}' under {local_root}. Provide a more specific path.\n{options}"
    )


def resolve_optional_local_model_reference(path_value: str | None, local_root: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None

    raw_value = path_value.strip()
    candidates = [raw_value]
    if not Path(raw_value).suffix:
        candidates.append(f"{raw_value}.pt")

    last_file_error: FileNotFoundError | None = None
    for candidate in candidates:
        try:
            resolved = resolve_local_artifact_reference(candidate, local_root)
        except FileNotFoundError as exc:
            last_file_error = exc
            continue
        if resolved.suffix.lower() == ".pt":
            return resolved

    if last_file_error is not None:
        return None
    return None


def discover_local_model_artifacts(local_root: Path, *, recursive: bool) -> list[tuple[Path, Path | None, str]]:
    candidates: list[tuple[float, str, Path, Path | None, str]] = []
    seen_model_paths: set[Path] = set()
    iterator = local_root.rglob("*.pt") if recursive else local_root.glob("*.pt")

    for model_path in iterator:
        if not model_path.is_file():
            continue

        resolved_model = model_path.resolve()
        if resolved_model in seen_model_paths:
            continue

        metadata_candidate = model_path.with_suffix(".yaml")
        resolved_metadata = metadata_candidate.resolve() if metadata_candidate.exists() and metadata_candidate.is_file() else None
        candidates.append(
            (model_path.stat().st_mtime, str(resolved_model).lower(), resolved_model, resolved_metadata, model_path.stem)
        )
        seen_model_paths.add(resolved_model)

    if not candidates:
        scope = "recursively under" if recursive else "directly under"
        raise FileNotFoundError(f"No .pt models found {scope}: {local_root}")

    candidates.sort(key=lambda item: (item[0], item[1]))
    return [(model_path, metadata_path, run_label) for _, _, model_path, metadata_path, run_label in candidates]


def select_local_artifacts_by_prefix(selector: str, local_root: Path) -> list[tuple[Path, Path | None, str]]:
    normalized = selector.strip().lower()
    if not normalized:
        return []

    prefixes = {normalized}
    if normalized.endswith(".pt"):
        prefixes.add(normalized[:-3])
    else:
        prefixes.add(f"{normalized}.pt")

    matches: list[tuple[Path, Path | None, str]] = []
    seen_model_paths: set[Path] = set()
    for model_path, metadata_path, run_label in discover_local_model_artifacts(local_root, recursive=True):
        model_name = model_path.name.lower()
        model_stem = model_path.stem.lower()
        if any(model_name.startswith(prefix) or model_stem.startswith(prefix.rstrip(".")) for prefix in prefixes):
            resolved_model = model_path.resolve()
            if resolved_model not in seen_model_paths:
                matches.append((resolved_model, metadata_path.resolve() if metadata_path else None, run_label))
                seen_model_paths.add(resolved_model)

    return matches


def resolve_latest_registry_reference(local_root: Path) -> Path:
    direct_candidate = (local_root / "exports" / "latest.yaml").resolve()
    if direct_candidate.exists():
        return direct_candidate

    matches = [
        candidate.resolve()
        for candidate in local_root.rglob("latest.yaml")
        if candidate.is_file() and candidate.parent.name == "exports"
    ]
    if not matches:
        raise FileNotFoundError(
            f"Missing latest registry under {local_root}. Expected exports/latest.yaml or a nested */exports/latest.yaml. "
            "Pass --model-path explicitly or export a model first."
        )

    if len(matches) == 1:
        return matches[0]

    local_matches = [candidate for candidate in matches if "local" in candidate.parts]
    if len(local_matches) == 1:
        return local_matches[0]

    ranked_matches = sorted(matches, key=lambda candidate: (candidate.stat().st_mtime, str(candidate).lower()))
    newest_match = ranked_matches[-1]
    newest_mtime = newest_match.stat().st_mtime
    newest_ties = [candidate for candidate in ranked_matches if candidate.stat().st_mtime == newest_mtime]
    if len(newest_ties) == 1:
        return newest_match

    options = "\n".join(f"- {candidate}" for candidate in matches[:12])
    raise ValueError(
        f"Ambiguous latest registry under {local_root}. Provide --model-path explicitly or narrow the local root.\n{options}"
    )


def resolve_with_project_root(path_value: str) -> Path:
    raw = Path(path_value)
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def is_all_run_selector(value: str | None) -> bool:
    return isinstance(value, str) and value.strip().lower() == ALL_RUN_LABEL_SELECTOR


def is_all_recursive_run_selector(value: str | None) -> bool:
    return isinstance(value, str) and value.strip().lower() in {ALL_RECURSIVE_RUN_LABEL_SELECTOR, "all-recursive"}


def is_bulk_run_selector(value: str | None) -> bool:
    return is_all_run_selector(value) or is_all_recursive_run_selector(value)


def parse_client_config_from_bundle(bundle: dict[str, Any]) -> dict[str, Any] | None:
    client_secrets = bundle.get("client_secrets")
    if isinstance(client_secrets, dict):
        return client_secrets

    installed = bundle.get("installed")
    if isinstance(installed, dict):
        return {"installed": installed}

    return None


def parse_token_info_from_bundle(bundle: dict[str, Any]) -> dict[str, Any] | None:
    token_info = bundle.get("token")
    if isinstance(token_info, dict):
        return token_info

    # Backward-compatible support for token-only JSON files.
    if isinstance(bundle.get("access_token"), str) or isinstance(bundle.get("token"), str):
        return bundle

    return None


def slug_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "-", value.strip())
    token = re.sub(r"-{2,}", "-", token).strip("-").lower()
    return token or "item"


def build_import_suffix(registry_name: str, run_label: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    return f"imported-from-{slug_token(registry_name)}-{slug_token(run_label)}-{timestamp}"


def resolve_import_target_path(base_path: Path, overwrite: bool, suffix: str) -> Path:
    if overwrite or not base_path.exists():
        return base_path

    candidate = base_path.with_name(f"{base_path.stem}__{suffix}{base_path.suffix}")
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = base_path.with_name(f"{base_path.stem}__{suffix}-{index}{base_path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def build_drive_service(
    *,
    oauth_credentials_file: Path,
    client_secrets_path: Path | None,
    token_path: Path | None,
):
    """Build an authenticated Google Drive API service using OAuth."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as ex:  # pragma: no cover - dependency guard
        raise ImportError(
            "Missing Google Drive OAuth dependencies. Install: "
            "google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        ) from ex

    bundle: dict[str, Any] = {}
    if oauth_credentials_file.exists():
        bundle = read_json(oauth_credentials_file)

    creds = None
    token_info = parse_token_info_from_bundle(bundle)
    if token_info:
        creds = Credentials.from_authorized_user_info(token_info, DRIVE_SCOPES)
    elif token_path is not None and token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), DRIVE_SCOPES)

    # If cached token was granted with narrower scopes, trigger a new OAuth consent.
    if creds and not creds.has_scopes(DRIVE_SCOPES):
        creds = None

    client_config = parse_client_config_from_bundle(bundle)
    if client_config is None and client_secrets_path is not None and client_secrets_path.exists():
        client_config = read_json(client_secrets_path)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if client_config is None:
                raise FileNotFoundError(
                    "OAuth credentials non trovate. Usa --oauth-credentials-file con un JSON che contiene "
                    "'client_secrets' oppure passa --oauth-client-secrets (legacy)."
                )

            flow = InstalledAppFlow.from_client_config(client_config, DRIVE_SCOPES)
            creds = flow.run_local_server(port=0)

    merged_bundle = dict(bundle)
    if client_config is not None and "client_secrets" not in merged_bundle:
        merged_bundle["client_secrets"] = client_config
    merged_bundle["token"] = json.loads(creds.to_json())
    write_json(oauth_credentials_file, merged_bundle)

    if token_path is not None:
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=creds)


def escape_drive_q(value: str) -> str:
    return value.replace("'", "\\'")


def find_drive_file_id(service, *, name: str, parent_id: str, mime_type: str | None = None) -> str | None:
    q_parts = [
        f"name = '{escape_drive_q(name)}'",
        f"'{parent_id}' in parents",
        "trashed = false",
    ]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")

    response = service.files().list(
        q=" and ".join(q_parts),
        spaces="drive",
        fields="files(id, name, modifiedTime)",
        pageSize=10,
        orderBy="modifiedTime desc",
    ).execute()
    files = response.get("files", [])
    if not files:
        return None
    return str(files[0]["id"])


def list_drive_files(service, *, parent_id: str, mime_type: str | None = None) -> list[dict[str, str]]:
    q_parts = [
        f"'{parent_id}' in parents",
        "trashed = false",
    ]
    if mime_type:
        q_parts.append(f"mimeType = '{mime_type}'")

    response = service.files().list(
        q=" and ".join(q_parts),
        spaces="drive",
        fields="files(id, name, modifiedTime)",
        pageSize=1000,
        orderBy="name",
    ).execute()
    files = response.get("files", [])
    return [file for file in files if isinstance(file, dict)]


def ensure_drive_folder(service, *, name: str, parent_id: str) -> str:
    folder_mime = "application/vnd.google-apps.folder"
    existing = find_drive_file_id(service, name=name, parent_id=parent_id, mime_type=folder_mime)
    if existing:
        return existing

    metadata = {"name": name, "mimeType": folder_mime, "parents": [parent_id]}
    created = service.files().create(body=metadata, fields="id").execute()
    return str(created["id"])


def _resolve_drive_child_folder_id(service, *, parent_id: str, selector: str) -> str:
    """Resolve a single Drive folder name under a specific parent.

    Matching order:
    1. exact case-sensitive name
    2. exact case-insensitive name
    3. prefix startswith (case-insensitive)
    4. contains substring (case-insensitive)
    """
    raw_selector = selector.strip()
    if not raw_selector:
        raise ValueError("Selector cartella vuoto")

    exact_id = find_drive_file_id(
        service,
        name=raw_selector,
        parent_id=parent_id,
        mime_type="application/vnd.google-apps.folder",
    )
    if exact_id:
        return exact_id

    folders = list_drive_files(
        service,
        parent_id=parent_id,
        mime_type="application/vnd.google-apps.folder",
    )
    if not folders:
        raise FileNotFoundError(f"Nessuna sottocartella trovata sotto parent '{parent_id}'")

    normalized = raw_selector.lower()

    exact_ci = [item for item in folders if str(item.get("name") or "").strip().lower() == normalized]
    if len(exact_ci) == 1:
        return str(exact_ci[0]["id"])
    if len(exact_ci) > 1:
        names = ", ".join(sorted(str(item.get("name") or "") for item in exact_ci))
        raise ValueError(f"Selector '{selector}' ambiguo (exact CI): {names}")

    prefix = [item for item in folders if str(item.get("name") or "").strip().lower().startswith(normalized)]
    if len(prefix) == 1:
        return str(prefix[0]["id"])
    if len(prefix) > 1:
        names = ", ".join(sorted(str(item.get("name") or "") for item in prefix)[:12])
        raise ValueError(f"Selector '{selector}' ambiguo (prefix): {names}")

    contains = [item for item in folders if normalized in str(item.get("name") or "").strip().lower()]
    if len(contains) == 1:
        return str(contains[0]["id"])
    if len(contains) > 1:
        names = ", ".join(sorted(str(item.get("name") or "") for item in contains)[:12])
        raise ValueError(f"Selector '{selector}' ambiguo (contains): {names}")

    available = ", ".join(sorted(str(item.get("name") or "") for item in folders)[:20])
    raise FileNotFoundError(
        f"Cartella Drive non trovata: {selector}. Cartelle disponibili sotto parent '{parent_id}': {available}"
    )


def resolve_drive_folder_id_by_selector(service, *, parent_id: str, selector: str) -> str:
    """Resolve a Drive folder selector, supporting nested paths like a/b/c."""
    raw_selector = selector.strip().strip("/")
    if not raw_selector:
        raise ValueError("Registry selector vuoto")

    if "/" in raw_selector:
        current_parent = parent_id
        for segment in [chunk.strip() for chunk in raw_selector.split("/") if chunk.strip()]:
            current_parent = _resolve_drive_child_folder_id(
                service,
                parent_id=current_parent,
                selector=segment,
            )
        return current_parent

    return _resolve_drive_child_folder_id(service, parent_id=parent_id, selector=raw_selector)


def upload_drive_file(service, *, parent_id: str, source_path: Path, target_name: str) -> str:
    from googleapiclient.http import MediaFileUpload

    existing = find_drive_file_id(service, name=target_name, parent_id=parent_id)
    media = MediaFileUpload(str(source_path), resumable=True)
    if existing:
        updated = service.files().update(fileId=existing, media_body=media, fields="id").execute()
        return str(updated["id"])

    metadata = {"name": target_name, "parents": [parent_id]}
    created = service.files().create(body=metadata, media_body=media, fields="id").execute()
    return str(created["id"])


def upload_drive_text(service, *, parent_id: str, filename: str, text_payload: str) -> str:
    from googleapiclient.http import MediaIoBaseUpload

    existing = find_drive_file_id(service, name=filename, parent_id=parent_id)
    stream = io.BytesIO(text_payload.encode("utf-8"))
    media = MediaIoBaseUpload(stream, mimetype="text/yaml", resumable=False)

    if existing:
        updated = service.files().update(fileId=existing, media_body=media, fields="id").execute()
        return str(updated["id"])

    metadata = {"name": filename, "parents": [parent_id]}
    created = service.files().create(body=metadata, media_body=media, fields="id").execute()
    return str(created["id"])


def download_drive_file(service, *, file_id: str, target_path: Path) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    target_path.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id)
    with target_path.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def read_drive_text(service, *, file_id: str) -> str:
    request = service.files().get_media(fileId=file_id)
    payload: bytes = request.execute()
    return payload.decode("utf-8")


def ordered_registry_run_labels(registry_root: Path) -> list[str]:
    models_root = registry_root / "models"
    if not models_root.exists():
        raise FileNotFoundError(f"Missing models folder in registry: {models_root}")

    run_labels = sorted(path.name for path in models_root.iterdir() if path.is_dir())
    if not run_labels:
        raise FileNotFoundError(f"No run folders found in registry: {models_root}")

    latest_path = registry_root / "latest.yaml"
    latest_run_label = ""
    if latest_path.exists():
        latest_payload = read_yaml(latest_path)
        latest_run_label = str(latest_payload.get("run_label") or "").strip()

    if latest_run_label and latest_run_label in run_labels:
        run_labels = [label for label in run_labels if label != latest_run_label] + [latest_run_label]

    return run_labels


def list_registry_entries(registry_root: Path) -> list[dict[str, str]]:
    models_root = registry_root / "models"
    if not models_root.exists():
        raise FileNotFoundError(f"Missing models folder in registry: {models_root}")

    entries: list[dict[str, str]] = []
    for run_dir in sorted(path for path in models_root.iterdir() if path.is_dir()):
        manifest_path = run_dir / "model_manifest.yaml"
        if not manifest_path.exists():
            continue
        manifest = read_yaml(manifest_path)
        model_filename = str(manifest.get("model_filename") or "").strip()
        if not model_filename:
            continue
        entries.append({"run_label": run_dir.name, "model_filename": model_filename})

    if not entries:
        raise FileNotFoundError(f"No importable runs found in registry: {models_root}")
    return entries


def list_flat_exports_entries(registry_root: Path) -> list[dict[str, str]]:
    """List importable entries from cloud layout: registry_root/exports/*.pt (+ optional .yaml)."""
    exports_root = registry_root / "exports"
    if not exports_root.exists():
        raise FileNotFoundError(f"Missing exports folder in registry: {exports_root}")

    entries: list[dict[str, str]] = []
    for model_path in sorted(exports_root.glob("*.pt")):
        if not model_path.is_file():
            continue
        metadata_path = model_path.with_suffix(".yaml")
        entries.append(
            {
                "run_label": model_path.stem,
                "model_filename": model_path.name,
                "metadata_filename": metadata_path.name if metadata_path.exists() and metadata_path.is_file() else "",
            }
        )

    if not entries:
        raise FileNotFoundError(f"No importable .pt models found in exports: {exports_root}")
    return entries


def resolve_flat_exports_entries_by_selector(selector: str, registry_root: Path) -> list[dict[str, str]]:
    """Resolve entries in flat exports layout by exact run/model name, then prefix startswith."""
    normalized = selector.strip().lower()
    if not normalized:
        return []

    entries = list_flat_exports_entries(registry_root)
    exact_matches = [
        entry
        for entry in entries
        if entry["run_label"].lower() == normalized
        or entry["model_filename"].lower() == normalized
        or Path(entry["model_filename"]).stem.lower() == normalized
    ]
    if exact_matches:
        return exact_matches

    prefixes = {normalized}
    if normalized.endswith(".pt"):
        prefixes.add(normalized[:-3])
    else:
        prefixes.add(f"{normalized}.pt")

    return [
        entry
        for entry in entries
        if any(
            entry["model_filename"].lower().startswith(prefix)
            or Path(entry["model_filename"]).stem.lower().startswith(prefix.rstrip("."))
            for prefix in prefixes
        )
    ]


def resolve_registry_run_labels_by_selector(selector: str, registry_root: Path) -> list[str]:
    normalized = selector.strip().lower()
    if not normalized:
        return []

    run_labels = ordered_registry_run_labels(registry_root)
    exact_run_matches = [run_label for run_label in run_labels if run_label.lower() == normalized]
    if exact_run_matches:
        return exact_run_matches

    entries = list_registry_entries(registry_root)
    exact_model_matches = [
        entry["run_label"]
        for entry in entries
        if entry["model_filename"].lower() == normalized or Path(entry["model_filename"]).stem.lower() == normalized
    ]
    if exact_model_matches:
        return [run_label for run_label in run_labels if run_label in exact_model_matches]

    prefixes = {normalized}
    if normalized.endswith(".pt"):
        prefixes.add(normalized[:-3])
    else:
        prefixes.add(f"{normalized}.pt")

    prefix_matches = [
        entry["run_label"]
        for entry in entries
        if any(
            entry["model_filename"].lower().startswith(prefix)
            or Path(entry["model_filename"]).stem.lower().startswith(prefix.rstrip("."))
            for prefix in prefixes
        )
    ]
    unique_prefix_matches: list[str] = []
    seen: set[str] = set()
    for run_label in run_labels:
        if run_label in prefix_matches and run_label not in seen:
            unique_prefix_matches.append(run_label)
            seen.add(run_label)
    return unique_prefix_matches


def ordered_drive_registry_run_labels(service, *, registry_id: str, models_id: str) -> list[str]:
    run_labels = sorted(
        str(item.get("name") or "").strip()
        for item in list_drive_files(
            service,
            parent_id=models_id,
            mime_type="application/vnd.google-apps.folder",
        )
    )
    run_labels = [label for label in run_labels if label]
    if not run_labels:
        raise FileNotFoundError("No run folders found in drive registry")

    latest_id = find_drive_file_id(service, name="latest.yaml", parent_id=registry_id)
    latest_run_label = ""
    if latest_id:
        latest_payload = yaml.safe_load(read_drive_text(service, file_id=latest_id)) or {}
        latest_run_label = str(latest_payload.get("run_label") or "").strip()

    if latest_run_label and latest_run_label in run_labels:
        run_labels = [label for label in run_labels if label != latest_run_label] + [latest_run_label]

    return run_labels


def list_drive_registry_entries(service, *, models_id: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for item in list_drive_files(
        service,
        parent_id=models_id,
        mime_type="application/vnd.google-apps.folder",
    ):
        run_label = str(item.get("name") or "").strip()
        run_dir_id = str(item.get("id") or "").strip()
        if not run_label or not run_dir_id:
            continue
        manifest_id = find_drive_file_id(service, name="model_manifest.yaml", parent_id=run_dir_id)
        if not manifest_id:
            continue
        manifest = yaml.safe_load(read_drive_text(service, file_id=manifest_id)) or {}
        model_filename = str(manifest.get("model_filename") or "").strip()
        if not model_filename:
            continue
        entries.append({"run_label": run_label, "model_filename": model_filename})

    if not entries:
        raise FileNotFoundError("No importable runs found in drive registry")
    return entries


def list_drive_flat_exports_entries(service, *, exports_id: str) -> list[dict[str, str]]:
    """List importable entries from drive cloud layout: exports/*.pt (+ optional .yaml)."""
    files = list_drive_files(service, parent_id=exports_id)
    by_name = {str(item.get("name") or ""): item for item in files}

    entries: list[dict[str, str]] = []
    for item in files:
        model_filename = str(item.get("name") or "").strip()
        if not model_filename.lower().endswith(".pt"):
            continue
        metadata_name = f"{Path(model_filename).stem}.yaml"
        entries.append(
            {
                "run_label": Path(model_filename).stem,
                "model_filename": model_filename,
                "metadata_filename": metadata_name if metadata_name in by_name else "",
            }
        )

    if not entries:
        raise FileNotFoundError("No importable .pt files found in drive exports folder")
    return entries


def resolve_drive_flat_exports_entries_by_selector(service, *, exports_id: str, selector: str) -> list[dict[str, str]]:
    """Resolve entries in drive flat exports layout by exact run/model name, then prefix startswith."""
    normalized = selector.strip().lower()
    if not normalized:
        return []

    entries = list_drive_flat_exports_entries(service, exports_id=exports_id)
    exact_matches = [
        entry
        for entry in entries
        if entry["run_label"].lower() == normalized
        or entry["model_filename"].lower() == normalized
        or Path(entry["model_filename"]).stem.lower() == normalized
    ]
    if exact_matches:
        return exact_matches

    prefixes = {normalized}
    if normalized.endswith(".pt"):
        prefixes.add(normalized[:-3])
    else:
        prefixes.add(f"{normalized}.pt")

    return [
        entry
        for entry in entries
        if any(
            entry["model_filename"].lower().startswith(prefix)
            or Path(entry["model_filename"]).stem.lower().startswith(prefix.rstrip("."))
            for prefix in prefixes
        )
    ]


def resolve_drive_registry_run_labels_by_selector(service, *, registry_id: str, models_id: str, selector: str) -> list[str]:
    normalized = selector.strip().lower()
    if not normalized:
        return []

    run_labels = ordered_drive_registry_run_labels(service, registry_id=registry_id, models_id=models_id)
    exact_run_matches = [run_label for run_label in run_labels if run_label.lower() == normalized]
    if exact_run_matches:
        return exact_run_matches

    entries = list_drive_registry_entries(service, models_id=models_id)
    exact_model_matches = [
        entry["run_label"]
        for entry in entries
        if entry["model_filename"].lower() == normalized or Path(entry["model_filename"]).stem.lower() == normalized
    ]
    if exact_model_matches:
        return [run_label for run_label in run_labels if run_label in exact_model_matches]

    prefixes = {normalized}
    if normalized.endswith(".pt"):
        prefixes.add(normalized[:-3])
    else:
        prefixes.add(f"{normalized}.pt")

    prefix_matches = [
        entry["run_label"]
        for entry in entries
        if any(
            entry["model_filename"].lower().startswith(prefix)
            or Path(entry["model_filename"]).stem.lower().startswith(prefix.rstrip("."))
            for prefix in prefixes
        )
    ]
    unique_prefix_matches: list[str] = []
    seen: set[str] = set()
    for run_label in run_labels:
        if run_label in prefix_matches and run_label not in seen:
            unique_prefix_matches.append(run_label)
            seen.add(run_label)
    return unique_prefix_matches


def choose_local_artifacts(
    *,
    local_persistent_root: Path,
    model_path_arg: str | None,
    metadata_path_arg: str | None,
    run_label_arg: str | None,
) -> tuple[Path, Path | None, str]:
    exports_root = local_persistent_root / "exports"

    if is_bulk_run_selector(run_label_arg):
        raise ValueError("bulk selectors are supported only at the command level")

    if model_path_arg:
        model_path = resolve_local_artifact_reference(model_path_arg, local_persistent_root)
        metadata_path = None
        if metadata_path_arg:
            metadata_path = resolve_local_artifact_reference(metadata_path_arg, local_persistent_root)
        else:
            metadata_candidate = model_path.with_suffix(".yaml")
            if metadata_candidate.exists() and metadata_candidate.is_file():
                metadata_path = metadata_candidate
        run_label = run_label_arg or model_path.stem
        return model_path, metadata_path, run_label

    run_label_model_path = resolve_optional_local_model_reference(run_label_arg, local_persistent_root)
    if run_label_model_path is not None:
        metadata_path = None
        metadata_candidate = run_label_model_path.with_suffix(".yaml")
        if metadata_candidate.exists() and metadata_candidate.is_file():
            metadata_path = metadata_candidate
        return run_label_model_path, metadata_path, run_label_model_path.stem

    version_dir: Path | None = None

    if run_label_arg:
        prefix_matches = select_local_artifacts_by_prefix(run_label_arg, local_persistent_root)
        if prefix_matches:
            if len(prefix_matches) == 1:
                return prefix_matches[0]
            raise ValueError(
                f"Selector '{run_label_arg}' matches multiple local models and must be handled at command level."
            )

    latest_path = resolve_latest_registry_reference(local_persistent_root)

    latest = read_yaml(latest_path)
    model_rel = latest.get("model_path")
    if not isinstance(model_rel, str) or not model_rel.strip():
        raise ValueError(f"latest.yaml does not contain a valid model_path: {latest_path}")

    metadata_rel = latest.get("metadata_path")
    run_label = run_label_arg or str(latest.get("run_label") or Path(model_rel).stem)

    model_path = resolve_local_artifact_reference(model_rel, local_persistent_root)
    metadata_path = None
    if isinstance(metadata_rel, str) and metadata_rel.strip():
        metadata_candidate = resolve_local_artifact_reference(metadata_rel, local_persistent_root)
        if metadata_candidate.exists():
            metadata_path = metadata_candidate

    return model_path, metadata_path, run_label


def export_to_drive(
    *,
    drive_root: Path,
    registry_name: str,
    local_persistent_root: Path,
    model_path_arg: str | None,
    metadata_path_arg: str | None,
    run_label_arg: str | None,
) -> None:
    if is_all_run_selector(run_label_arg) or is_all_recursive_run_selector(run_label_arg):
        if model_path_arg or metadata_path_arg:
            raise ValueError("--run-label all cannot be combined with --model-path or --metadata-path")

        artifacts = discover_local_model_artifacts(
            local_persistent_root,
            recursive=is_all_recursive_run_selector(run_label_arg),
        )
        print(f"Discovered {len(artifacts)} model(s) under {local_persistent_root}")
        for model_path, metadata_path, run_label in artifacts:
            export_to_drive(
                drive_root=drive_root,
                registry_name=registry_name,
                local_persistent_root=local_persistent_root,
                model_path_arg=str(model_path),
                metadata_path_arg=(str(metadata_path) if metadata_path is not None else None),
                run_label_arg=run_label,
            )
        return

    if run_label_arg and not model_path_arg:
        prefix_matches = select_local_artifacts_by_prefix(run_label_arg, local_persistent_root)
        if len(prefix_matches) > 1:
            print(f"Selector '{run_label_arg}' matched {len(prefix_matches)} local model(s)")
            for model_path, metadata_path, run_label in prefix_matches:
                export_to_drive(
                    drive_root=drive_root,
                    registry_name=registry_name,
                    local_persistent_root=local_persistent_root,
                    model_path_arg=str(model_path),
                    metadata_path_arg=(str(metadata_path) if metadata_path is not None else None),
                    run_label_arg=run_label,
                )
            return

    model_path, metadata_path, run_label = choose_local_artifacts(
        local_persistent_root=local_persistent_root,
        model_path_arg=model_path_arg,
        metadata_path_arg=metadata_path_arg,
        run_label_arg=run_label_arg,
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if metadata_path is not None and not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    registry_root = (drive_root / registry_name).resolve()
    version_dir = registry_root / "models" / run_label
    version_dir.mkdir(parents=True, exist_ok=True)

    target_model = version_dir / model_path.name
    shutil.copy2(model_path, target_model)

    target_metadata: Path | None = None
    if metadata_path is not None:
        target_metadata = version_dir / metadata_path.name
        shutil.copy2(metadata_path, target_metadata)

    model_hash = sha256_of_file(target_model)
    model_size = target_model.stat().st_size

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "model_filename": target_model.name,
        "metadata_filename": target_metadata.name if target_metadata else None,
        "model_sha256": model_hash,
        "model_size_bytes": model_size,
    }
    write_yaml(version_dir / "model_manifest.yaml", manifest)

    latest = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "manifest_path": f"models/{run_label}/model_manifest.yaml",
    }
    write_yaml(registry_root / "latest.yaml", latest)

    print(f"Export completed: {target_model}")
    print(f"SHA256: {model_hash}")
    print(f"Drive latest: {registry_root / 'latest.yaml'}")


def export_to_drive_oauth(
    *,
    service,
    drive_parent_id: str,
    registry_name: str,
    local_persistent_root: Path,
    model_path_arg: str | None,
    metadata_path_arg: str | None,
    run_label_arg: str | None,
) -> None:
    if is_all_run_selector(run_label_arg) or is_all_recursive_run_selector(run_label_arg):
        if model_path_arg or metadata_path_arg:
            raise ValueError("--run-label all cannot be combined with --model-path or --metadata-path")

        artifacts = discover_local_model_artifacts(
            local_persistent_root,
            recursive=is_all_recursive_run_selector(run_label_arg),
        )
        print(f"Discovered {len(artifacts)} model(s) under {local_persistent_root}")
        for model_path, metadata_path, run_label in artifacts:
            export_to_drive_oauth(
                service=service,
                drive_parent_id=drive_parent_id,
                registry_name=registry_name,
                local_persistent_root=local_persistent_root,
                model_path_arg=str(model_path),
                metadata_path_arg=(str(metadata_path) if metadata_path is not None else None),
                run_label_arg=run_label,
            )
        return

    if run_label_arg and not model_path_arg:
        prefix_matches = select_local_artifacts_by_prefix(run_label_arg, local_persistent_root)
        if len(prefix_matches) > 1:
            print(f"Selector '{run_label_arg}' matched {len(prefix_matches)} local model(s)")
            for model_path, metadata_path, run_label in prefix_matches:
                export_to_drive_oauth(
                    service=service,
                    drive_parent_id=drive_parent_id,
                    registry_name=registry_name,
                    local_persistent_root=local_persistent_root,
                    model_path_arg=str(model_path),
                    metadata_path_arg=(str(metadata_path) if metadata_path is not None else None),
                    run_label_arg=run_label,
                )
            return

    model_path, metadata_path, run_label = choose_local_artifacts(
        local_persistent_root=local_persistent_root,
        model_path_arg=model_path_arg,
        metadata_path_arg=metadata_path_arg,
        run_label_arg=run_label_arg,
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if metadata_path is not None and not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    try:
        registry_id = resolve_drive_folder_id_by_selector(
            service,
            parent_id=drive_parent_id,
            selector=registry_name,
        )
    except FileNotFoundError:
        registry_id = ensure_drive_folder(service, name=registry_name, parent_id=drive_parent_id)
    models_id = ensure_drive_folder(service, name="models", parent_id=registry_id)
    run_dir_id = ensure_drive_folder(service, name=run_label, parent_id=models_id)

    upload_drive_file(service, parent_id=run_dir_id, source_path=model_path, target_name=model_path.name)
    if metadata_path is not None:
        upload_drive_file(service, parent_id=run_dir_id, source_path=metadata_path, target_name=metadata_path.name)

    model_hash = sha256_of_file(model_path)
    model_size = model_path.stat().st_size
    manifest_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "model_filename": model_path.name,
        "metadata_filename": metadata_path.name if metadata_path else None,
        "model_sha256": model_hash,
        "model_size_bytes": model_size,
    }
    upload_drive_text(
        service,
        parent_id=run_dir_id,
        filename="model_manifest.yaml",
        text_payload=yaml.safe_dump(manifest_payload, sort_keys=False, allow_unicode=False),
    )

    latest_payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "manifest_path": f"models/{run_label}/model_manifest.yaml",
    }
    upload_drive_text(
        service,
        parent_id=registry_id,
        filename="latest.yaml",
        text_payload=yaml.safe_dump(latest_payload, sort_keys=False, allow_unicode=False),
    )

    print(f"Export completed via OAuth for run_label={run_label}")
    print(f"SHA256: {model_hash}")


def import_from_drive(
    *,
    drive_root: Path,
    registry_name: str,
    target_persistent_root: Path,
    run_label_arg: str | None,
    overwrite: bool,
) -> None:
    registry_root = (drive_root / registry_name).resolve()
    if not registry_root.exists():
        raise FileNotFoundError(f"Drive registry folder not found: {registry_root}")

    models_root = registry_root / "models"
    flat_exports_root = registry_root / "exports"
    use_flat_exports_layout = not models_root.exists() and flat_exports_root.exists()
    version_dir: Path | None = None

    if is_bulk_run_selector(run_label_arg):
        if use_flat_exports_layout:
            entries = list_flat_exports_entries(registry_root)
            print(f"Importing {len(entries)} export file(s) from flat registry layout {registry_name}")
            for entry in entries:
                import_from_drive(
                    drive_root=drive_root,
                    registry_name=registry_name,
                    target_persistent_root=target_persistent_root,
                    run_label_arg=entry["run_label"],
                    overwrite=overwrite,
                )
            return

        run_labels = ordered_registry_run_labels(registry_root)
        print(f"Importing {len(run_labels)} run(s) from registry {registry_name}")
        for run_label in run_labels:
            import_from_drive(
                drive_root=drive_root,
                registry_name=registry_name,
                target_persistent_root=target_persistent_root,
                run_label_arg=run_label,
                overwrite=overwrite,
            )
        return

    if use_flat_exports_layout:
        if run_label_arg:
            matching_entries = resolve_flat_exports_entries_by_selector(run_label_arg, registry_root)
            if matching_entries:
                if len(matching_entries) > 1:
                    print(
                        f"Selector '{run_label_arg}' matched {len(matching_entries)} export file(s) in flat registry {registry_name}"
                    )
                    for entry in matching_entries:
                        import_from_drive(
                            drive_root=drive_root,
                            registry_name=registry_name,
                            target_persistent_root=target_persistent_root,
                            run_label_arg=entry["run_label"],
                            overwrite=overwrite,
                        )
                    return
                selected = matching_entries[0]
            else:
                raise FileNotFoundError(
                    f"No matching model for selector '{run_label_arg}' in flat exports registry: {flat_exports_root}"
                )
        else:
            latest_path = registry_root / "latest.yaml"
            if latest_path.exists():
                latest = read_yaml(latest_path)
                model_ref = str(latest.get("model_path") or "").strip()
                if not model_ref:
                    raise ValueError("latest.yaml does not contain model_path")
                model_filename = Path(model_ref).name
                selected = {
                    "run_label": str(latest.get("run_label") or Path(model_filename).stem),
                    "model_filename": model_filename,
                    "metadata_filename": Path(str(latest.get("metadata_path") or "")).name,
                }
            else:
                entries = list_flat_exports_entries(registry_root)
                selected = entries[-1]

        run_label = str(selected["run_label"])
        model_filename = str(selected["model_filename"])
        metadata_filename = str(selected.get("metadata_filename") or "")
        source_model = flat_exports_root / model_filename
        if not source_model.exists():
            raise FileNotFoundError(f"Model file not found in flat exports: {source_model}")

        actual_sha = sha256_of_file(source_model)
    else:
        if run_label_arg:
            matching_run_labels = resolve_registry_run_labels_by_selector(run_label_arg, registry_root)
            if matching_run_labels:
                if len(matching_run_labels) > 1:
                    print(f"Selector '{run_label_arg}' matched {len(matching_run_labels)} run(s) in registry {registry_name}")
                    for run_label in matching_run_labels:
                        import_from_drive(
                            drive_root=drive_root,
                            registry_name=registry_name,
                            target_persistent_root=target_persistent_root,
                            run_label_arg=run_label,
                            overwrite=overwrite,
                        )
                    return
                run_label = matching_run_labels[0]
            else:
                run_label = run_label_arg
        else:
            latest = read_yaml(registry_root / "latest.yaml")
            run_label = str(latest.get("run_label") or "").strip()
            if not run_label:
                raise ValueError("latest.yaml does not contain run_label")

        version_dir = registry_root / "models" / run_label
        manifest_path = version_dir / "model_manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing model manifest: {manifest_path}")

        manifest = read_yaml(manifest_path)
        model_filename = str(manifest.get("model_filename") or "").strip()
        metadata_filename = str(manifest.get("metadata_filename") or "").strip()
        expected_sha = str(manifest.get("model_sha256") or "").strip()
        if not model_filename:
            raise ValueError(f"Invalid model_filename in {manifest_path}")

        source_model = version_dir / model_filename
        if not source_model.exists():
            raise FileNotFoundError(f"Model file not found on drive: {source_model}")

        actual_sha = sha256_of_file(source_model)
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(f"SHA256 mismatch for {source_model}. expected={expected_sha} actual={actual_sha}")

    target_root = target_persistent_root
    target_root.mkdir(parents=True, exist_ok=True)

    import_suffix = build_import_suffix(registry_name, run_label)

    target_model = resolve_import_target_path(target_root / model_filename, overwrite, import_suffix)
    shutil.copy2(source_model, target_model)

    target_metadata: Path | None = None
    if metadata_filename:
        if use_flat_exports_layout:
            source_metadata = flat_exports_root / metadata_filename
        else:
            assert version_dir is not None
            source_metadata = version_dir / metadata_filename
        if source_metadata.exists():
            target_metadata = resolve_import_target_path(target_root / metadata_filename, overwrite, import_suffix)
            shutil.copy2(source_metadata, target_metadata)

    latest_local = {
        "run_label": run_label,
        "model_path": target_model.name,
        "metadata_path": target_metadata.name if target_metadata else None,
    }
    write_yaml(target_root / "latest.yaml", latest_local)

    if target_model.name != model_filename:
        print(f"Model name conflict detected, saved as: {target_model.name}")
    if target_metadata is not None and target_metadata.name != metadata_filename:
        print(f"Metadata name conflict detected, saved as: {target_metadata.name}")

    print(f"Import completed: {target_model}")
    print(f"SHA256 verified: {actual_sha}")
    print(f"Local latest: {target_root / 'latest.yaml'}")


def import_from_drive_oauth(
    *,
    service,
    drive_parent_id: str,
    registry_name: str,
    target_persistent_root: Path,
    run_label_arg: str | None,
    overwrite: bool,
) -> None:
    registry_id = resolve_drive_folder_id_by_selector(
        service,
        parent_id=drive_parent_id,
        selector=registry_name,
    )

    models_id = find_drive_file_id(
        service,
        name="models",
        parent_id=registry_id,
        mime_type="application/vnd.google-apps.folder",
    )
    exports_id = find_drive_file_id(
        service,
        name="exports",
        parent_id=registry_id,
        mime_type="application/vnd.google-apps.folder",
    )
    use_flat_exports_layout = not models_id and bool(exports_id)
    if not models_id and not use_flat_exports_layout:
        raise FileNotFoundError("Neither models nor exports folder found in drive registry")
    models_id_str = str(models_id) if models_id is not None else ""

    if is_bulk_run_selector(run_label_arg):
        if use_flat_exports_layout:
            assert exports_id is not None
            entries = list_drive_flat_exports_entries(service, exports_id=exports_id)
            print(f"Importing {len(entries)} export file(s) from flat drive registry {registry_name}")
            for entry in entries:
                import_from_drive_oauth(
                    service=service,
                    drive_parent_id=drive_parent_id,
                    registry_name=registry_name,
                    target_persistent_root=target_persistent_root,
                    run_label_arg=entry["run_label"],
                    overwrite=overwrite,
                )
            return

        run_labels = ordered_drive_registry_run_labels(service, registry_id=registry_id, models_id=models_id_str)
        print(f"Importing {len(run_labels)} run(s) from drive registry {registry_name}")
        for run_label in run_labels:
            import_from_drive_oauth(
                service=service,
                drive_parent_id=drive_parent_id,
                registry_name=registry_name,
                target_persistent_root=target_persistent_root,
                run_label_arg=run_label,
                overwrite=overwrite,
            )
        return

    if use_flat_exports_layout:
        assert exports_id is not None
        if run_label_arg:
            matching_entries = resolve_drive_flat_exports_entries_by_selector(
                service,
                exports_id=exports_id,
                selector=run_label_arg,
            )
            if matching_entries:
                if len(matching_entries) > 1:
                    print(
                        f"Selector '{run_label_arg}' matched {len(matching_entries)} export file(s) in flat drive registry {registry_name}"
                    )
                    for entry in matching_entries:
                        import_from_drive_oauth(
                            service=service,
                            drive_parent_id=drive_parent_id,
                            registry_name=registry_name,
                            target_persistent_root=target_persistent_root,
                            run_label_arg=entry["run_label"],
                            overwrite=overwrite,
                        )
                    return
                selected = matching_entries[0]
            else:
                raise FileNotFoundError(
                    f"No matching model for selector '{run_label_arg}' in flat drive exports for registry {registry_name}"
                )
        else:
            latest_id = find_drive_file_id(service, name="latest.yaml", parent_id=registry_id)
            if latest_id:
                latest = yaml.safe_load(read_drive_text(service, file_id=latest_id)) or {}
                model_ref = str(latest.get("model_path") or "").strip()
                if not model_ref:
                    raise ValueError("latest.yaml does not contain model_path")
                model_filename = Path(model_ref).name
                selected = {
                    "run_label": str(latest.get("run_label") or Path(model_filename).stem),
                    "model_filename": model_filename,
                    "metadata_filename": Path(str(latest.get("metadata_path") or "")).name,
                }
            else:
                entries = list_drive_flat_exports_entries(service, exports_id=exports_id)
                selected = entries[-1]

        run_label = str(selected["run_label"])
        model_filename = str(selected["model_filename"])
        metadata_filename = str(selected.get("metadata_filename") or "")
        expected_sha = ""

        model_id = find_drive_file_id(service, name=model_filename, parent_id=exports_id)
        if not model_id:
            raise FileNotFoundError(f"Model file not found in drive flat exports for run {run_label}: {model_filename}")
    else:
        if run_label_arg:
            matching_run_labels = resolve_drive_registry_run_labels_by_selector(
                service,
                registry_id=registry_id,
                models_id=models_id_str,
                selector=run_label_arg,
            )
            if matching_run_labels:
                if len(matching_run_labels) > 1:
                    print(f"Selector '{run_label_arg}' matched {len(matching_run_labels)} run(s) in drive registry {registry_name}")
                    for run_label in matching_run_labels:
                        import_from_drive_oauth(
                            service=service,
                            drive_parent_id=drive_parent_id,
                            registry_name=registry_name,
                            target_persistent_root=target_persistent_root,
                            run_label_arg=run_label,
                            overwrite=overwrite,
                        )
                    return
                run_label = matching_run_labels[0]
            else:
                run_label = run_label_arg
        else:
            latest_id = find_drive_file_id(service, name="latest.yaml", parent_id=registry_id)
            if not latest_id:
                raise FileNotFoundError("latest.yaml not found in drive registry")
            latest = yaml.safe_load(read_drive_text(service, file_id=latest_id)) or {}
            run_label = str(latest.get("run_label") or "").strip()
            if not run_label:
                raise ValueError("latest.yaml does not contain run_label")

        run_dir_id = find_drive_file_id(
            service,
            name=run_label,
            parent_id=models_id_str,
            mime_type="application/vnd.google-apps.folder",
        )
        if not run_dir_id:
            raise FileNotFoundError(f"Run folder not found in drive registry: {run_label}")
        run_dir_id_str = str(run_dir_id)

        manifest_id = find_drive_file_id(service, name="model_manifest.yaml", parent_id=run_dir_id_str)
        if not manifest_id:
            raise FileNotFoundError(f"model_manifest.yaml not found for run: {run_label}")

        manifest = yaml.safe_load(read_drive_text(service, file_id=manifest_id)) or {}
        model_filename = str(manifest.get("model_filename") or "").strip()
        metadata_filename = str(manifest.get("metadata_filename") or "").strip()
        expected_sha = str(manifest.get("model_sha256") or "").strip()
        if not model_filename:
            raise ValueError("Invalid model_filename in model_manifest.yaml")

        model_id = find_drive_file_id(service, name=model_filename, parent_id=run_dir_id_str)
        if not model_id:
            raise FileNotFoundError(f"Model file not found on drive for run {run_label}: {model_filename}")

    target_root = target_persistent_root
    target_root.mkdir(parents=True, exist_ok=True)
    import_suffix = build_import_suffix(registry_name, run_label)

    target_model = resolve_import_target_path(target_root / model_filename, overwrite, import_suffix)
    download_drive_file(service, file_id=model_id, target_path=target_model)

    actual_sha = sha256_of_file(target_model)
    if expected_sha and actual_sha != expected_sha:
        target_model.unlink(missing_ok=True)
        raise ValueError(
            f"SHA256 mismatch for imported model. expected={expected_sha} actual={actual_sha}"
        )

    target_metadata: Path | None = None
    if metadata_filename:
        if use_flat_exports_layout:
            assert exports_id is not None
            metadata_parent_id = exports_id
        else:
            metadata_parent_id = run_dir_id_str
        metadata_id = find_drive_file_id(service, name=metadata_filename, parent_id=metadata_parent_id)
        if metadata_id:
            target_metadata = resolve_import_target_path(target_root / metadata_filename, overwrite, import_suffix)
            download_drive_file(service, file_id=metadata_id, target_path=target_metadata)

    latest_local = {
        "run_label": run_label,
        "model_path": target_model.name,
        "metadata_path": target_metadata.name if target_metadata else None,
    }
    write_yaml(target_root / "latest.yaml", latest_local)

    if target_model.name != model_filename:
        print(f"Model name conflict detected, saved as: {target_model.name}")
    if target_metadata is not None and target_metadata.name != metadata_filename:
        print(f"Metadata name conflict detected, saved as: {target_metadata.name}")

    print(f"Import completed: {target_model}")
    print(f"SHA256 verified: {actual_sha}")
    print(f"Local latest: {target_root / 'latest.yaml'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync model artifacts with Google Drive (filesystem or OAuth)")
    parser.add_argument("command", choices=["export", "import"], help="Operation to execute")
    parser.add_argument(
        "--auth-mode",
        choices=["filesystem", "oauth"],
        default="filesystem",
        help="Drive access mode: local synced folder or OAuth API",
    )
    parser.add_argument(
        "--drive-root",
        type=str,
        default=os.environ.get("GOOGLE_DRIVE_ROOT", ""),
        help="Google Drive local root (filesystem mode only, ex: G:/My Drive).",
    )
    parser.add_argument(
        "--drive-parent-id",
        type=str,
        default=os.environ.get("GOOGLE_DRIVE_PARENT_ID", "root"),
        help="OAuth mode only: parent folder ID in Google Drive (default: root)",
    )
    parser.add_argument(
        "--oauth-credentials-file",
        type=str,
        default=DEFAULT_OAUTH_CREDENTIALS_FILE,
        help=(
            "OAuth mode: single local JSON file containing both 'client_secrets' "
            "and generated 'token'"
        ),
    )
    parser.add_argument(
        "--oauth-client-secrets",
        type=str,
        default="",
        help="OAuth mode legacy: path to Google client secrets JSON",
    )
    parser.add_argument(
        "--oauth-token-path",
        type=str,
        default="",
        help="OAuth mode legacy: optional token JSON path for backward compatibility",
    )
    parser.add_argument(
        "--registry-name",
        type=str,
        default="yolo-fire-detector-models",
        help="Subfolder name used as model registry inside drive root",
    )

    parser.add_argument(
        "--local-persistent-root",
        type=str,
        default="artifacts/local",
        help="Local persistent root used for export discovery",
    )
    parser.add_argument("--model-path", type=str, default="", help="Model path for export (optional)")
    parser.add_argument("--metadata-path", type=str, default="", help="Metadata path for export (optional)")
    parser.add_argument(
        "--run-label",
        type=str,
        nargs="?",
        const="",
        default="",
        help="Run label for export/import (optional)",
    )

    parser.add_argument(
        "--target-persistent-root",
        type=str,
        default="artifacts/local",
        help="Target local persistent root used for import",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite local files on import")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_label = args.run_label.strip() or None
    model_path = args.model_path.strip() or None
    metadata_path = args.metadata_path.strip() or None

    if args.auth_mode == "filesystem":
        drive_root_raw = args.drive_root.strip()
        if not drive_root_raw:
            raise ValueError("Filesystem mode requires --drive-root (or GOOGLE_DRIVE_ROOT)")

        drive_root = Path(drive_root_raw).resolve()
        if not drive_root.exists():
            raise FileNotFoundError(f"Drive root not found: {drive_root}")

        if args.command == "export":
            export_to_drive(
                drive_root=drive_root,
                registry_name=args.registry_name,
                local_persistent_root=(PROJECT_ROOT / args.local_persistent_root).resolve(),
                model_path_arg=model_path,
                metadata_path_arg=metadata_path,
                run_label_arg=run_label,
            )
            return

        import_from_drive(
            drive_root=drive_root,
            registry_name=args.registry_name,
            target_persistent_root=(PROJECT_ROOT / args.target_persistent_root).resolve(),
            run_label_arg=run_label,
            overwrite=args.overwrite,
        )
        return

    client_secrets_arg = args.oauth_client_secrets.strip()
    token_path_arg = args.oauth_token_path.strip()

    service = build_drive_service(
        oauth_credentials_file=resolve_with_project_root(args.oauth_credentials_file),
        client_secrets_path=(resolve_with_project_root(client_secrets_arg) if client_secrets_arg else None),
        token_path=(resolve_with_project_root(token_path_arg) if token_path_arg else None),
    )
    drive_parent_id = args.drive_parent_id.strip() or "root"

    if args.command == "export":
        export_to_drive_oauth(
            service=service,
            drive_parent_id=drive_parent_id,
            registry_name=args.registry_name,
            local_persistent_root=(PROJECT_ROOT / args.local_persistent_root).resolve(),
            model_path_arg=model_path,
            metadata_path_arg=metadata_path,
            run_label_arg=run_label,
        )
        return

    import_from_drive_oauth(
        service=service,
        drive_parent_id=drive_parent_id,
        registry_name=args.registry_name,
        target_persistent_root=(PROJECT_ROOT / args.target_persistent_root).resolve(),
        run_label_arg=run_label,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

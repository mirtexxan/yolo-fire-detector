"""Download themed background photos from Unsplash for pseudo-real domains."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from urllib import parse, request


UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"
# WARNING: hardcoded on explicit user request for a private repository.
# Unsplash API uses the Access Key as Client-ID for search/download endpoints.
HARDCODED_UNSPLASH_ACCESS_KEY = "fgBldLSOgms2bJKvpsky70dofXBZ5d5N3sN6GZ0iBCk"


def load_unsplash_access_key(project_root: Path | None = None) -> str:
    """Resolve Unsplash access key from env, then .env.local/.env, then hardcoded fallback."""
    key = os.getenv("UNSPLASH_ACCESS_KEY", "").strip()
    if key:
        return key

    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    for env_name in (".env.local", ".env"):
        env_path = root / env_name
        if not env_path.exists() or not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() != "UNSPLASH_ACCESS_KEY":
                continue
            parsed = value.strip().strip('"').strip("'")
            if parsed:
                return parsed

    return HARDCODED_UNSPLASH_ACCESS_KEY


def _http_get_json(url: str, *, headers: dict[str, str], timeout: int = 30) -> dict:
    req = request.Request(url=url, headers=headers, method="GET")
    with request.urlopen(req, timeout=timeout) as resp:
        content = resp.read().decode("utf-8")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("Risposta API non valida")
    return payload


def _download_file(url: str, destination: Path, *, timeout: int = 60) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    req = request.Request(url=url, method="GET")
    with request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    destination.write_bytes(data)


def fetch_backgrounds(
    *,
    access_key: str,
    themes: list[str],
    output_root: Path,
    total_per_theme: int,
    per_page: int,
    orientation: str,
    min_width: int,
) -> dict:
    headers = {
        "Authorization": f"Client-ID {access_key}",
        "Accept-Version": "v1",
    }

    summary: dict[str, dict[str, int | str]] = {}

    for theme in themes:
        theme_slug = theme.strip().lower().replace(" ", "-")
        theme_dir = output_root / theme_slug
        theme_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        page = 1
        seen_ids: set[str] = set()

        while downloaded < total_per_theme:
            query = parse.urlencode(
                {
                    "query": theme,
                    "page": page,
                    "per_page": min(30, per_page),
                    "orientation": orientation,
                    "content_filter": "high",
                }
            )
            payload = _http_get_json(f"{UNSPLASH_SEARCH_URL}?{query}", headers=headers)
            results = payload.get("results", [])
            if not isinstance(results, list) or not results:
                break

            for item in results:
                if downloaded >= total_per_theme:
                    break
                if not isinstance(item, dict):
                    continue

                photo_id = str(item.get("id", "")).strip()
                if not photo_id or photo_id in seen_ids:
                    continue

                width = int(item.get("width", 0) or 0)
                if width < min_width:
                    continue

                urls = item.get("urls", {})
                if not isinstance(urls, dict):
                    continue
                image_url = str(urls.get("regular", "")).strip() or str(urls.get("full", "")).strip()
                if not image_url:
                    continue

                file_name = f"{downloaded + 1:04d}_{photo_id}.jpg"
                target_path = theme_dir / file_name
                try:
                    _download_file(image_url, target_path)
                except Exception:
                    continue

                seen_ids.add(photo_id)
                downloaded += 1

            page += 1

        summary[theme_slug] = {
            "requested": total_per_theme,
            "downloaded": downloaded,
            "output": theme_dir.as_posix(),
        }

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source": "unsplash",
        "themes": summary,
    }
    manifest_path = output_root / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scarica sfondi da Unsplash per domini pseudo-reali")
    parser.add_argument("--themes", type=str, required=True, help="Lista temi separati da virgola, es: forest,industrial,kitchen")
    parser.add_argument("--count", type=int, default=120, help="Numero immagini per tema")
    parser.add_argument("--per-page", type=int, default=30, help="Risultati per pagina API")
    parser.add_argument("--orientation", type=str, default="landscape", choices=["landscape", "portrait", "squarish"])
    parser.add_argument("--min-width", type=int, default=1200, help="Larghezza minima immagine")
    parser.add_argument(
        "--output-root",
        type=str,
        default="artifacts/local/background_domains/unsplash",
        help="Cartella di output",
    )
    parser.add_argument(
        "--access-key",
        type=str,
        default="",
        help="Unsplash Access Key (alternativa: env UNSPLASH_ACCESS_KEY)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    access_key = args.access_key.strip() or load_unsplash_access_key()
    if not access_key:
        raise ValueError("Manca Access Key Unsplash. Usa --access-key, env UNSPLASH_ACCESS_KEY, .env.local o .env")

    themes = [t.strip() for t in args.themes.split(",") if t.strip()]
    if not themes:
        raise ValueError("Specifica almeno un tema in --themes")

    manifest = fetch_backgrounds(
        access_key=access_key,
        themes=themes,
        output_root=Path(args.output_root).resolve(),
        total_per_theme=max(1, args.count),
        per_page=max(1, args.per_page),
        orientation=args.orientation,
        min_width=max(1, args.min_width),
    )

    print("Download completato")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

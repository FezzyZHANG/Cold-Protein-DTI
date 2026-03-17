#!/usr/bin/env python
"""Download pretrained ESM checkpoints to a local staging directory."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import urllib.request


MODEL_MANIFEST = {
    "esm2_t12_35M_UR50D": {
        "url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt",
        "filename": "esm2_t12_35M_UR50D.pt",
        "description": "Small ESM2 checkpoint for quick validation runs.",
    },
    "esm2_t33_650M_UR50D": {
        "url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
        "filename": "esm2_t33_650M_UR50D.pt",
        "description": "Recommended ESM2 checkpoint for the cold-protein experiment scaffold.",
    },
}


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def download_file(url: str, destination: Path, proxy: str | None, force: bool) -> None:
    if destination.exists() and not force:
        print(f"[download] file already exists, skipping: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)

    with opener.open(url) as response, destination.open("wb") as handle:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                print(f"[download] {format_size(downloaded)} / {format_size(total)}", end="\r")
            else:
                print(f"[download] {format_size(downloaded)}", end="\r")
    print()


def write_metadata(output_path: Path, model_name: str, url: str) -> None:
    metadata = {
        "model_name": model_name,
        "path": str(output_path),
        "source_url": url,
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a pretrained ESM checkpoint for local staging.")
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model key to download. Repeat to download multiple entries.",
    )
    parser.add_argument("--output-dir", default="artifacts/pretrained/esm")
    parser.add_argument("--proxy", default=None, help="Optional HTTP/HTTPS proxy, e.g. http://127.0.0.1:7890")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing local file.")
    parser.add_argument("--show-models", action="store_true", help="Print available manifest entries and exit.")
    args = parser.parse_args()

    if args.show_models:
        for name, info in MODEL_MANIFEST.items():
            print(f"{name}: {info['description']}")
            print(f"  {info['url']}")
        return

    models = args.model or ["esm2_t33_650M_UR50D"]
    output_dir = Path(args.output_dir)

    for model_name in models:
        if model_name not in MODEL_MANIFEST:
            available = ", ".join(sorted(MODEL_MANIFEST))
            raise SystemExit(f"Unknown model '{model_name}'. Available choices: {available}")

        info = MODEL_MANIFEST[model_name]
        output_path = output_dir / info["filename"]
        print(f"[download] model={model_name}")
        print(f"[download] source={info['url']}")
        print(f"[download] target={output_path}")
        download_file(info["url"], output_path, proxy=args.proxy, force=bool(args.force))
        write_metadata(output_path, model_name=model_name, url=info["url"])
        print(f"[download] completed: {output_path}")


if __name__ == "__main__":
    main()


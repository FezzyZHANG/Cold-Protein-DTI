#!/usr/bin/env python
"""Download pretrained ESM checkpoints to a local staging directory."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any
import urllib.request


MODEL_MANIFEST = {
    "esm2_t12_35M_UR50D": {
        "base_url": "https://huggingface.co/facebook/esm2_t12_35M_UR50D/resolve/main",
        "files": [
            "config.json",
            "tokenizer_config.json",
            "vocab.txt",
            "README.md",
            "model.safetensors",
        ],
        "description": "Small ESM2 checkpoint for quick validation runs.",
    },
    "esm2_t33_650M_UR50D": {
        "base_url": "https://huggingface.co/facebook/esm2_t33_650M_UR50D/resolve/main",
        "files": [
            "config.json",
            "tokenizer_config.json",
            "vocab.txt",
            "README.md",
            "model.safetensors",
        ],
        "description": "Recommended ESM2 checkpoint for the cold-protein experiment scaffold.",
    },
    "esmc_600m": {
        "files": [
            {
                "path": "README.md",
                "url": "https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12/resolve/main/README.md",
            },
            {
                "path": "config.json",
                "url": "https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12/resolve/main/config.json",
            },
            {
                "path": "data/weights/esmc_600m_2024_12_v0.pth",
                "url": (
                    "https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12/resolve/main/"
                    "data/weights/esmc_600m_2024_12_v0.pth"
                ),
            },
        ],
        "description": "ESM C 600M checkpoint for the optional `esmc` backend.",
    },
    "VESM_650M": {
        "files": [
            {
                "path": "README.md",
                "url": "https://huggingface.co/ntranoslab/vesm/resolve/main/README.md",
            },
            {
                "path": "VESM_650M.pth",
                "url": "https://huggingface.co/ntranoslab/vesm/resolve/main/VESM_650M.pth",
            },
        ],
        "description": (
            "VESM 650M overlay checkpoint. Runtime also needs the staged "
            "`esm2_t33_650M_UR50D` base model."
        ),
        "requires": ["esm2_t33_650M_UR50D"],
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


def iter_manifest_files(info: dict[str, Any]) -> list[tuple[str, str]]:
    if "base_url" in info:
        return [(filename, f"{info['base_url']}/{filename}") for filename in info["files"]]
    return [(item["path"], item["url"]) for item in info["files"]]


def write_metadata(output_dir: Path, model_name: str, source_urls: list[str]) -> None:
    metadata = {
        "model_name": model_name,
        "path": str(output_dir),
        "source_urls": source_urls,
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "download.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a pretrained ESM checkpoint for local staging.")
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model key to download. Repeat to download multiple entries.",
    )
    parser.add_argument("--output-dir", default="artifacts/pretrained")
    parser.add_argument("--proxy", default=None, help="Optional HTTP/HTTPS proxy, e.g. http://127.0.0.1:7890")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing local file.")
    parser.add_argument("--show-models", action="store_true", help="Print available manifest entries and exit.")
    args = parser.parse_args()

    if args.show_models:
        for name, info in MODEL_MANIFEST.items():
            print(f"{name}: {info['description']}")
            if "base_url" in info:
                print(f"  {info['base_url']}")
            for requirement in info.get("requires", []):
                print(f"  requires: {requirement}")
        return

    models = args.model or ["esm2_t33_650M_UR50D"]
    output_dir = Path(args.output_dir)

    for model_name in models:
        if model_name not in MODEL_MANIFEST:
            available = ", ".join(sorted(MODEL_MANIFEST))
            raise SystemExit(f"Unknown model '{model_name}'. Available choices: {available}")

        info = MODEL_MANIFEST[model_name]
        model_output_dir = output_dir / model_name
        manifest_files = iter_manifest_files(info)
        source_urls = [source_url for _, source_url in manifest_files]
        print(f"[download] model={model_name}")
        print(f"[download] target={model_output_dir}")
        for requirement in info.get("requires", []):
            required_path = output_dir / requirement
            if not required_path.exists():
                print(
                    "[download] note: "
                    f"{model_name} expects staged dependency {required_path}. "
                    f"Download it with `--model {requirement}` if it is not already present."
                )
        for relative_path, source_url in manifest_files:
            output_path = model_output_dir / relative_path
            print(f"[download] source={source_url}")
            print(f"[download] file={output_path}")
            download_file(source_url, output_path, proxy=args.proxy, force=bool(args.force))
        write_metadata(model_output_dir, model_name=model_name, source_urls=source_urls)
        print(f"[download] completed: {model_output_dir}")


if __name__ == "__main__":
    main()

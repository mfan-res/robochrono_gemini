#!/usr/bin/env python3
# coding: utf-8
"""Download the gated RoboChrono API split from Hugging Face.

Examples:
    python tools/download_robochrono_api.py --all
    python tools/download_robochrono_api.py v20_q4 v21_q4
    python tools/download_robochrono_api.py --all --qa-only
    python tools/download_robochrono_api.py v20_q4 --media-types clips

The source repository is:
https://huggingface.co/datasets/GIM-RoboLab/robochrono/tree/main/api

This dataset is gated. Log in first with ``hf auth login`` /
``huggingface-cli login`` or set ``HF_TOKEN`` after accepting the dataset
terms on the Hugging Face page.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "GIM-RoboLab/robochrono"
SOURCE_PREFIX = "api"
MEDIA_TYPES = ("clips", "episodes", "frames")
TRANSIENT_RETRIES = 3


def _require_hf():
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        sys.exit(
            "huggingface_hub is not installed. Install it with:\n"
            "  pip install huggingface_hub"
        )
    return HfApi, hf_hub_download, GatedRepoError, RepositoryNotFoundError


def _token(explicit: str | None) -> str | None:
    return explicit or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _is_fatal(exc: Exception) -> bool:
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        RepositoryNotFoundError,
    )

    if isinstance(exc, (EntryNotFoundError, GatedRepoError, RepositoryNotFoundError)):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (401, 403, 404)


def list_api_files(repo_id: str, token: str | None) -> list[str]:
    HfApi, _, GatedRepoError, RepositoryNotFoundError = _require_hf()
    try:
        files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        raise SystemExit(_access_help(repo_id, exc)) from exc
    return sorted(f for f in files if f.startswith(f"{SOURCE_PREFIX}/"))


def discover_versions(files: list[str]) -> dict[str, list[str]]:
    versions: dict[str, list[str]] = {}
    for path in files:
        parts = path.split("/")
        if len(parts) < 3:
            continue
        versions.setdefault(parts[1], []).append(path)
    return versions


def select_files(
    versions: dict[str, list[str]],
    wanted: list[str],
    qa_only: bool,
    media_types: list[str],
) -> list[str]:
    selected: list[str] = []
    media_prefixes = tuple(f"{SOURCE_PREFIX}/{{version}}/media/{kind}/" for kind in media_types)

    for version in wanted:
        for path in versions[version]:
            manifest = path == f"{SOURCE_PREFIX}/{version}/manifest.json"
            qa = path.startswith(f"{SOURCE_PREFIX}/{version}/qa/")
            media = (not qa_only) and any(
                path.startswith(prefix.format(version=version)) for prefix in media_prefixes
            )
            if manifest or qa or media:
                selected.append(path)
    return selected


def _access_help(repo_id: str, exc: Exception) -> str:
    return (
        f"Cannot access {repo_id}: {type(exc).__name__}.\n"
        "This is a gated Hugging Face dataset. Open the dataset page, accept "
        "the access terms, then run one of:\n"
        "  hf auth login\n"
        "  huggingface-cli login\n"
        "or set HF_TOKEN and rerun this script."
    )


def download_one(repo_id: str, repo_path: str, out: Path, token: str | None) -> None:
    _, hf_hub_download, _, _ = _require_hf()
    rel = Path(*Path(repo_path).parts[1:])
    dest = out / rel
    if dest.exists() and dest.stat().st_size > 0:
        return

    for attempt in range(1, TRANSIENT_RETRIES + 1):
        try:
            cached = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=repo_path,
                    token=token,
                )
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".incomplete")
            shutil.copy2(cached, tmp)
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001 - classified immediately below
            if _is_fatal(exc):
                raise SystemExit(_access_help(repo_id, exc)) from exc
            if attempt == TRANSIENT_RETRIES:
                raise
            wait = 15 * attempt
            print(
                f"transient failure on {repo_path} ({type(exc).__name__}); "
                f"retry {attempt}/{TRANSIENT_RETRIES - 1} in {wait}s",
                flush=True,
            )
            time.sleep(wait)


def validate(out: Path) -> int:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "robochrono",
            "validate-data",
            "--data-root",
            str(out.resolve()),
        ],
        cwd=REPO_ROOT,
    ).returncode


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download GIM-RoboLab/robochrono api split from Hugging Face."
    )
    ap.add_argument("versions", nargs="*", help="versions to download, e.g. v20_q4")
    ap.add_argument("--all", action="store_true", help="download every api version")
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "scenarios",
        help="local data root (default: scenarios/)",
    )
    ap.add_argument("--repo-id", default=REPO_ID, help=f"Hugging Face dataset repo (default: {REPO_ID})")
    ap.add_argument("--token", help="Hugging Face token; defaults to HF_TOKEN or login cache")
    ap.add_argument("--list", action="store_true", help="list available versions and exit")
    ap.add_argument("--qa-only", action="store_true", help="download manifest and qa only, no media")
    ap.add_argument(
        "--media-types",
        default=",".join(MEDIA_TYPES),
        help="comma-separated subset of clips,episodes,frames",
    )
    ap.add_argument("--skip-validate", action="store_true", help="do not run validate-data afterwards")
    args = ap.parse_args()

    token = _token(args.token)
    files = list_api_files(args.repo_id, token)
    versions = discover_versions(files)
    available = sorted(versions)

    if args.list:
        for version in available:
            print(f"{version}\t{len(versions[version])} files")
        return 0

    if bool(args.versions) == args.all:
        ap.error("name versions or pass --all, one or the other")

    media_types = [] if args.qa_only else [k.strip() for k in args.media_types.split(",") if k.strip()]
    unknown = sorted(set(media_types) - set(MEDIA_TYPES))
    if unknown:
        ap.error(f"unknown media types {unknown}; known: {list(MEDIA_TYPES)}")

    wanted = available if args.all else args.versions
    missing = sorted(set(wanted) - set(available))
    if missing:
        sys.exit(f"not found under api/: {missing}\navailable: {available}")

    selected = select_files(versions, wanted, args.qa_only, media_types)
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"downloading {len(selected)} file(s) from {args.repo_id}/api "
        f"to {args.output.resolve()}",
        flush=True,
    )

    for i, repo_path in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {repo_path}", flush=True)
        download_one(args.repo_id, repo_path, args.output, token)

    if args.skip_validate:
        return 0
    if args.qa_only or set(media_types) != set(MEDIA_TYPES):
        print("\nnote: media was skipped or filtered; validation may report missing files.", flush=True)
    print("\nrunning validate-data:", flush=True)
    return validate(args.output)


if __name__ == "__main__":
    sys.exit(main())

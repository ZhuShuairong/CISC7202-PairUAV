#!/usr/bin/env python3
"""Prepare PairUAV data using the official UAVM baseline workflow.

Official flow parity:
1) unzip University-Release.zip
2) copy University-Release/train/drone/* -> train_tour/
3) remove University-Release and University-Release.zip
4) download YaxuanLi/UAVM_2026_test dataset archives
5) extract train.tar, test.tar, test_tour.tar
6) remove tar files and .cache

Example:
    python scripts/prepare_pairuav_data.py --workdir /root/autodl-tmp/university/PairUAV
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

DEFAULT_DATASET_REPO = "YaxuanLi/UAVM_2026_test"
REQUIRED_ARCHIVES = ("train.tar", "test.tar", "test_tour.tar")


def emit(message: str) -> None:
    print(message, flush=True)


def resolve_university_root(workdir: Path) -> Path | None:
    candidates = [
        workdir / "University-Release",
        workdir / "University-Release" / "University-Release",
    ]
    for root in candidates:
        if (root / "train" / "drone").is_dir():
            return root
    return None


def maybe_unzip_university_release(workdir: Path, university_zip: Path) -> Path:
    root = resolve_university_root(workdir)
    if root is not None:
        emit(f"[prep] Found University-Release at {root}")
        return root

    if not university_zip.is_file():
        raise FileNotFoundError(
            "University-Release not found and zip is missing. "
            f"Expected extracted folder under {workdir} or zip file {university_zip}."
        )

    emit(f"[prep] Extracting {university_zip} ...")
    with zipfile.ZipFile(university_zip, "r") as archive:
        archive.extractall(workdir)

    root = resolve_university_root(workdir)
    if root is None:
        raise FileNotFoundError(
            "Unable to locate University-Release/train/drone after unzip. "
            "Please verify the archive structure."
        )

    emit(f"[prep] Extracted University-Release to {root}")
    return root


def copy_train_tour(university_root: Path, workdir: Path) -> int:
    drone_root = university_root / "train" / "drone"
    if not drone_root.is_dir():
        raise FileNotFoundError(f"Missing expected training drone folder: {drone_root}")

    train_tour = workdir / "train_tour"
    train_tour.mkdir(parents=True, exist_ok=True)

    copied_items = 0
    for item in sorted(drone_root.iterdir(), key=lambda path: path.name.lower()):
        target = train_tour / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
        copied_items += 1

    emit(f"[prep] Copied {copied_items} entries from {drone_root} -> {train_tour}")
    return copied_items


def _run_download_command(command: list[str], workdir: Path, token: str | None) -> None:
    env = os.environ.copy()
    if token:
        env["HF_TOKEN"] = token
        env["HUGGINGFACE_HUB_TOKEN"] = token

    emit("[prep] Running: " + " ".join(command))
    subprocess.run(command, cwd=str(workdir), check=True, env=env)


def _download_archives_with_hf(workdir: Path, repo_id: str, token: str | None) -> None:
    if shutil.which("hf") is None:
        raise RuntimeError("hf CLI was not found")
    _run_download_command(
        ["hf", "download", "--repo-type", "dataset", repo_id, "--local-dir", str(workdir)],
        workdir,
        token,
    )


def _download_archives_with_huggingface_cli(workdir: Path, repo_id: str, token: str | None) -> None:
    if shutil.which("huggingface-cli") is None:
        raise RuntimeError("huggingface-cli was not found")
    _run_download_command(
        ["huggingface-cli", "download", "--repo-type", "dataset", repo_id, "--local-dir", str(workdir)],
        workdir,
        token,
    )


def _download_archives_with_python(workdir: Path, repo_id: str, token: str | None) -> None:
    try:
        import importlib
        snapshot_download = importlib.import_module("huggingface_hub").snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("huggingface_hub is not installed") from exc

    emit("[prep] Downloading archives with huggingface_hub.snapshot_download ...")
    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "local_dir": str(workdir),
        "allow_patterns": list(REQUIRED_ARCHIVES),
    }
    if token:
        kwargs["token"] = token

    try:
        snapshot_download(local_dir_use_symlinks=False, **kwargs)
    except TypeError:
        # Support older/newer API signatures.
        snapshot_download(**kwargs)


def archive_paths(workdir: Path) -> list[Path]:
    return [workdir / name for name in REQUIRED_ARCHIVES]


def has_required_archives(workdir: Path) -> bool:
    return all(path.is_file() for path in archive_paths(workdir))


def download_archives(workdir: Path, repo_id: str, token: str | None, tool: str) -> str:
    if has_required_archives(workdir):
        emit("[prep] Required archives already exist; skipping download")
        return "skipped"

    tools = [tool] if tool != "auto" else ["hf", "huggingface-cli", "python"]
    errors: list[str] = []

    for candidate in tools:
        try:
            if candidate == "hf":
                _download_archives_with_hf(workdir, repo_id, token)
            elif candidate == "huggingface-cli":
                _download_archives_with_huggingface_cli(workdir, repo_id, token)
            elif candidate == "python":
                _download_archives_with_python(workdir, repo_id, token)
            else:
                raise ValueError(f"Unsupported download tool: {candidate}")

            if has_required_archives(workdir):
                emit(f"[prep] Download completed using {candidate}")
                return candidate
            errors.append(f"{candidate}: download command finished but archives are still missing")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate}: {exc}")

    raise RuntimeError("Failed to download required archives. " + " | ".join(errors))


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        member_target = (destination / member.name).resolve()
        if os.path.commonpath([str(destination), str(member_target)]) != str(destination):
            raise RuntimeError(f"Blocked unsafe tar entry: {member.name}")
    archive.extractall(destination)


def extract_archives(workdir: Path) -> None:
    for archive_path in archive_paths(workdir):
        if not archive_path.is_file():
            raise FileNotFoundError(f"Missing archive after download: {archive_path}")
        emit(f"[prep] Extracting {archive_path.name} ...")
        with tarfile.open(archive_path, "r") as archive:
            _safe_extract_tar(archive, workdir)


def cleanup(
    workdir: Path,
    university_zip: Path,
    keep_university_release: bool,
    keep_university_zip: bool,
    keep_archives: bool,
    keep_cache: bool,
) -> None:
    if not keep_university_release:
        extracted_dir = workdir / "University-Release"
        if extracted_dir.exists():
            emit(f"[prep] Removing {extracted_dir}")
            shutil.rmtree(extracted_dir)

    if not keep_university_zip and university_zip.exists():
        if university_zip.is_relative_to(workdir):
            emit(f"[prep] Removing {university_zip}")
            university_zip.unlink()
        else:
            emit(
                "[prep] Skipping zip cleanup because it is outside workdir: "
                f"{university_zip}"
            )

    if not keep_archives:
        for path in archive_paths(workdir):
            if path.exists():
                emit(f"[prep] Removing {path}")
                path.unlink()

    if not keep_cache:
        cache_dir = workdir / ".cache"
        if cache_dir.exists():
            emit(f"[prep] Removing {cache_dir}")
            shutil.rmtree(cache_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PairUAV data with official baseline workflow")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="Directory where data is prepared (contains University-Release.zip and output folders)",
    )
    parser.add_argument(
        "--university-zip",
        type=Path,
        default=None,
        help="Path to University-Release.zip (defaults to <workdir>/University-Release.zip)",
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default=DEFAULT_DATASET_REPO,
        help="Hugging Face dataset repo ID for competition archives",
    )
    parser.add_argument(
        "--download-tool",
        choices=["auto", "hf", "huggingface-cli", "python"],
        default="auto",
        help="Archive download backend",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token (falls back to HF_TOKEN/HUGGINGFACE_HUB_TOKEN env vars)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Hugging Face download step (archives must already exist)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip tar extraction step",
    )
    parser.add_argument(
        "--keep-university-release",
        action="store_true",
        help="Keep extracted University-Release folder instead of deleting it",
    )
    parser.add_argument(
        "--keep-university-zip",
        action="store_true",
        help="Keep University-Release.zip instead of deleting it",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep train.tar/test.tar/test_tour.tar after extraction",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep .cache directory after download",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = args.workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    if args.university_zip:
        university_zip = args.university_zip.expanduser()
        if not university_zip.is_absolute():
            university_zip = workdir / university_zip
        university_zip = university_zip.resolve()
    else:
        university_zip = workdir / "University-Release.zip"
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    emit("[prep] PairUAV official-style preparation started")
    emit(f"[prep] Workdir: {workdir}")

    university_root = maybe_unzip_university_release(workdir, university_zip)
    copy_train_tour(university_root, workdir)

    if args.skip_download:
        emit("[prep] Skip download enabled")
        if not has_required_archives(workdir) and not args.skip_extract:
            raise FileNotFoundError(
                "skip-download was set but required archives are missing. "
                f"Expected: {', '.join(REQUIRED_ARCHIVES)} in {workdir}."
            )
    else:
        download_archives(workdir, args.dataset_repo, token, args.download_tool)

    if args.skip_extract:
        emit("[prep] Skip extract enabled")
    else:
        extract_archives(workdir)

    cleanup(
        workdir,
        university_zip,
        keep_university_release=args.keep_university_release,
        keep_university_zip=args.keep_university_zip,
        keep_archives=args.keep_archives,
        keep_cache=args.keep_cache,
    )

    emit("[prep] Done")
    emit(
        "[prep] Created/updated folders: "
        f"{workdir / 'train_tour'}, {workdir / 'train'}, {workdir / 'test'}, {workdir / 'test_tour'}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[prep] Interrupted", file=sys.stderr)
        raise

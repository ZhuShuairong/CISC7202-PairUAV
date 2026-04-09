"""Generate CodaBench submission files for PairUAV with strict order parity checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from PIL import Image
import torch
import torchvision.models as models

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset_pairuav import build_geometry_features
from models.baseline import siamese_fusion
from models.geopairnet import GeoPairNet
from models.harp_dual_path import HARPDualPath
from models.harp_pose_lite import HARPPoseLite


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PAIR_MANIFEST_NAMES = (
    "pairs.txt",
    "pair.txt",
    "pair_list.txt",
    "test_pairs.txt",
    "query_pairs.txt",
    "pairs.csv",
    "test.csv",
    "annotations.csv",
)

DEFAULT_PAIRUAV_ROOT_CANDIDATES = (
    Path("/root/autodl-tmp/university/University-Release/University-Release"),
    Path("/root/autodl-tmp/university/PairUAV"),
    Path("/root/autodl-tmp/university/PairUAV-Processed"),
    Path("/root/autodl-pub/PairUAV"),
)


PairOrderMode = Literal["official", "auto"]
DelimiterMode = Literal["comma", "space"]


@dataclass(frozen=True)
class PairEntry:
    source: Path
    target: Path
    source_ref: str
    target_ref: str
    pair_id: str


def _normalize_ref(ref: str) -> str:
    return ref.strip().replace("\\", "/").lstrip("./")


def _natural_sort_key(text: str) -> tuple[Any, ...]:
    pieces = re.split(r"(\d+)", text.lower())
    key: list[Any] = []
    for piece in pieces:
        if piece.isdigit():
            key.append(int(piece))
        else:
            key.append(piece)
    return tuple(key)


def _pair_identifier(index: int, source_ref: str, target_ref: str) -> str:
    return f"{index:06d}:{source_ref}||{target_ref}"


def resolve_pairuav_root(root: str | None) -> Path:
    checked: list[Path] = []
    seen: set[str] = set()
    candidates: list[Path] = []

    for env_name in ("PAIRUAV_ROOT", "PAIRUAV_DATA_ROOT", "PAIRUAV_PROCESSED_ROOT"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())

    if root:
        explicit = Path(root).expanduser()
        checked.append(explicit)
        seen.add(str(explicit))
        if explicit.is_dir():
            return explicit.resolve()

    candidates.extend(DEFAULT_PAIRUAV_ROOT_CANDIDATES)

    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        checked.append(candidate)
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not determine a usable PairUAV root. Checked: "
        + ", ".join(str(path) for path in checked)
        + ". Set --pairuav-root or PAIRUAV_ROOT."
    )


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _scan_image_files(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.rglob("*") if _is_image_file(path)],
        key=lambda path: _natural_sort_key(str(path.relative_to(root)).replace("\\", "/")),
    )


def _build_image_index(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]], list[Path]]:
    images = _scan_image_files(root)
    by_relative: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = {}
    for path in images:
        relative = str(path.relative_to(root)).replace("\\", "/")
        by_relative[relative] = path
        by_name.setdefault(path.name.lower(), []).append(path)
    return by_relative, by_name, images


def _resolve_image_ref(
    ref: str,
    root: Path,
    by_relative: dict[str, Path] | None,
    by_name: dict[str, list[Path]] | None,
) -> Path | None:
    candidate = Path(ref)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    normalized = _normalize_ref(ref)
    direct = root / normalized
    if direct.exists():
        return direct

    for prefix in ("test/", "test_tour/", "train/", "train_tour/"):
        prefixed = root / prefix / normalized
        if prefixed.exists():
            return prefixed

    if by_relative is not None and normalized in by_relative:
        return by_relative[normalized]

    if by_name is not None:
        matches = by_name.get(Path(normalized).name.lower(), [])
        if len(matches) == 1:
            return matches[0]

    return None


def _parse_manifest(
    manifest: Path,
    root: Path,
    by_relative: dict[str, Path] | None,
    by_name: dict[str, list[Path]] | None,
) -> tuple[list[PairEntry], bool]:
    pairs: list[PairEntry] = []
    had_unresolved = False

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "," in line:
                fields = [field.strip() for field in next(csv.reader([line]))]
            elif "\t" in line:
                fields = [field.strip() for field in line.split("\t")]
            else:
                fields = line.split()

            if len(fields) < 2:
                continue

            source_ref = _normalize_ref(fields[0])
            target_ref = _normalize_ref(fields[1])
            source = _resolve_image_ref(source_ref, root, by_relative, by_name)
            target = _resolve_image_ref(target_ref, root, by_relative, by_name)
            if source is None or target is None:
                had_unresolved = True
                continue

            pair_index = len(pairs) + 1
            pairs.append(
                PairEntry(
                    source=source,
                    target=target,
                    source_ref=source_ref,
                    target_ref=target_ref,
                    pair_id=_pair_identifier(pair_index, source_ref, target_ref),
                )
            )

    return pairs, had_unresolved


def _collect_official_json_pairs(
    root: Path,
    by_relative: dict[str, Path] | None,
    by_name: dict[str, list[Path]] | None,
) -> tuple[list[PairEntry], bool]:
    test_root = root / "test"
    if not test_root.is_dir():
        return [], False

    json_paths = sorted(
        [path for path in test_root.rglob("*.json") if path.is_file()],
        key=lambda path: _natural_sort_key(str(path.relative_to(test_root)).replace("\\", "/")),
    )
    pairs: list[PairEntry] = []
    had_unresolved = False

    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        source_raw = payload.get("image_a") or payload.get("source")
        target_raw = payload.get("image_b") or payload.get("target")
        if not source_raw or not target_raw:
            continue

        source_ref = _normalize_ref(str(source_raw))
        target_ref = _normalize_ref(str(target_raw))
        source = _resolve_image_ref(source_ref, root, by_relative, by_name)
        target = _resolve_image_ref(target_ref, root, by_relative, by_name)
        if source is None or target is None:
            had_unresolved = True
            continue

        pair_index = len(pairs) + 1
        pairs.append(
            PairEntry(
                source=source,
                target=target,
                source_ref=source_ref,
                target_ref=target_ref,
                pair_id=_pair_identifier(pair_index, source_ref, target_ref),
            )
        )

    return pairs, had_unresolved


def _group_image_sets(directory: Path) -> dict[str, list[Path]]:
    subdirs = [path for path in directory.iterdir() if path.is_dir()]
    if not subdirs:
        images = _scan_image_files(directory)
        return {"": images} if images else {}

    grouped: dict[str, list[Path]] = {}
    for subdir in sorted(subdirs, key=lambda path: _natural_sort_key(path.name)):
        grouped[subdir.name] = _scan_image_files(subdir)
    return grouped


def _find_split_pair_dirs(root: Path) -> tuple[Path, Path] | None:
    split_roots = [root / "test", root / "test_tour", root]
    directory_name_pairs = [
        ("query_drone", "gallery_drone"),
        ("query", "gallery"),
        ("source_drone", "target_drone"),
        ("source", "target"),
    ]

    for split_root in split_roots:
        for query_name, gallery_name in directory_name_pairs:
            query_dir = split_root / query_name
            gallery_dir = split_root / gallery_name
            if query_dir.is_dir() and gallery_dir.is_dir():
                return query_dir, gallery_dir
    return None


def _discover_pairs(
    root: Path,
    pair_order: PairOrderMode = "official",
) -> tuple[list[PairEntry], str]:
    if pair_order == "official":
        fast_json_pairs, fast_json_unresolved = _collect_official_json_pairs(root, None, None)
        if fast_json_pairs and not fast_json_unresolved:
            return fast_json_pairs, "official:test_json"

        for base in (root / "test", root / "test_tour", root):
            for name in PAIR_MANIFEST_NAMES:
                candidate = base / name
                if candidate.is_file():
                    pairs, unresolved = _parse_manifest(candidate, root, None, None)
                    if pairs and not unresolved:
                        return pairs, f"official:manifest:{candidate}"

        by_relative, by_name, _ = _build_image_index(root)

        json_pairs, _ = _collect_official_json_pairs(root, by_relative, by_name)
        if json_pairs:
            return json_pairs, "official:test_json"

        for base in (root / "test", root / "test_tour", root):
            for name in PAIR_MANIFEST_NAMES:
                candidate = base / name
                if candidate.is_file():
                    pairs, _ = _parse_manifest(candidate, root, by_relative, by_name)
                    if pairs:
                        return pairs, f"official:manifest:{candidate}"

        raise FileNotFoundError(
            "Official pair-order mode requires test JSON pair annotations or a pair manifest under the processed "
            f"PairUAV root, but none were found in {root}."
        )

    by_relative, by_name, _ = _build_image_index(root)

    split_dirs = _find_split_pair_dirs(root)
    if split_dirs is None:
        raise FileNotFoundError(
            f"Could not find test pair data under {root}. "
            "Expected a manifest file, test JSON pairs, or query/gallery split directories."
        )

    query_dir, gallery_dir = split_dirs
    query_groups = _group_image_sets(query_dir)
    gallery_groups = _group_image_sets(gallery_dir)

    fallback_pairs: list[PairEntry] = []
    common_group_names = [name for name in sorted(query_groups) if name in gallery_groups]

    if common_group_names:
        for group_name in common_group_names:
            query_images = query_groups[group_name]
            gallery_images = gallery_groups[group_name]
            if not query_images or not gallery_images:
                continue
            pair_count = min(len(query_images), len(gallery_images))
            for query_image, gallery_image in zip(query_images[:pair_count], gallery_images[:pair_count]):
                pair_index = len(fallback_pairs) + 1
                fallback_pairs.append(
                    PairEntry(
                        source=query_image,
                        target=gallery_image,
                        source_ref=str(query_image.relative_to(root)).replace("\\", "/"),
                        target_ref=str(gallery_image.relative_to(root)).replace("\\", "/"),
                        pair_id=_pair_identifier(
                            pair_index,
                            str(query_image.relative_to(root)).replace("\\", "/"),
                            str(gallery_image.relative_to(root)).replace("\\", "/"),
                        ),
                    )
                )
        if fallback_pairs:
            return fallback_pairs, "fallback:grouped_split_dirs"

    query_flat = _scan_image_files(query_dir)
    gallery_flat = _scan_image_files(gallery_dir)
    pair_count = min(len(query_flat), len(gallery_flat))

    for query_image, gallery_image in zip(query_flat[:pair_count], gallery_flat[:pair_count]):
        pair_index = len(fallback_pairs) + 1
        fallback_pairs.append(
            PairEntry(
                source=query_image,
                target=gallery_image,
                source_ref=str(query_image.relative_to(root)).replace("\\", "/"),
                target_ref=str(gallery_image.relative_to(root)).replace("\\", "/"),
                pair_id=_pair_identifier(
                    pair_index,
                    str(query_image.relative_to(root)).replace("\\", "/"),
                    str(gallery_image.relative_to(root)).replace("\\", "/"),
                ),
            )
        )

    return fallback_pairs, "fallback:flat_split_dirs"


def _load_model(checkpoint: str, device: torch.device) -> tuple[torch.nn.Module, str]:
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model_config = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}

    geopair_keys = any(
        key.startswith("rotation_head.")
        or key.startswith("distance_head.")
        or key.startswith("global_projector.")
        for key in state_dict.keys()
    )
    dual_path_keys = any(
        key.startswith("wide.") or key.startswith("deep.") or key.startswith("fusion.")
        for key in state_dict.keys()
    )
    pose_lite_keys = any(
        key.startswith("regression_head.") or key.startswith("conf_head.")
        for key in state_dict.keys()
    )

    if geopair_keys:
        model = GeoPairNet(**model_config).to(device)
        model.load_state_dict(state_dict, strict=False)
        return model.eval(), "geopairnet"

    if dual_path_keys and not pose_lite_keys:
        model = HARPDualPath(frozen=True, use_gate=True)
        model.phase = 2
        model.load_state_dict(state_dict, strict=False)
        return model.eval().to(device), "dual_path"

    if pose_lite_keys:
        model = HARPPoseLite().to(device)
        model.load_state_dict(state_dict, strict=False)
        return model.eval(), "pose_lite"

    try:
        model = HARPPoseLite().to(device)
        model.load_state_dict(state_dict, strict=False)
        return model.eval(), "pose_lite"
    except Exception:
        model = HARPDualPath(frozen=True, use_gate=True)
        model.phase = 2
        model.load_state_dict(state_dict, strict=False)
        return model.eval().to(device), "dual_path"


def _feature_cache_path(cache_root: Path, root: Path, image_path: Path) -> Path:
    try:
        relative = image_path.relative_to(root)
    except ValueError:
        digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:16]
        relative = Path("external") / digest / image_path.name
    return (cache_root / relative).with_suffix(".pt")


def _extract_features(
    model: torch.nn.Module,
    model_kind: str,
    image_paths: Sequence[Path],
    device: torch.device,
    root: Path,
    cache_dir: str | None = None,
    batch_size: int = 32,
) -> dict[Path, Any]:
    transform = models.ResNet50_Weights.IMAGENET1K_V1.transforms()

    cache_root = Path(cache_dir) if cache_dir else None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)

    features: dict[Path, Any] = {}
    pending_paths: list[Path] = []

    for path in image_paths:
        if cache_root is not None:
            cache_file = _feature_cache_path(cache_root, root, path)
            if cache_file.is_file():
                loaded = torch.load(cache_file, map_location="cpu")
                features[path] = loaded
                continue
        pending_paths.append(path)

    backbone = None
    if model_kind != "geopairnet":
        backbone = model.backbone  # type: ignore[attr-defined]
        backbone.eval()

    for start in range(0, len(pending_paths), batch_size):
        batch_paths = list(pending_paths[start:start + batch_size])
        batch = torch.stack([transform(Image.open(path).convert("RGB")) for path in batch_paths]).to(device)

        global_embed = None
        spatial_embed = None
        batch_features = None
        with torch.inference_mode():
            if model_kind == "geopairnet":
                global_embed, spatial_embed = model.encode(batch)  # type: ignore[attr-defined]
            else:
                assert backbone is not None
                batch_features = backbone(batch)

        if model_kind == "geopairnet":
            assert global_embed is not None and spatial_embed is not None
            for idx, path in enumerate(batch_paths):
                stored = {
                    "global": global_embed[idx].detach().cpu().half(),
                    "spatial": spatial_embed[idx].detach().cpu().half(),
                }
                features[path] = stored
                if cache_root is not None:
                    cache_file = _feature_cache_path(cache_root, root, path)
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(stored, cache_file)
        else:
            assert batch_features is not None
            for path, feat in zip(batch_paths, batch_features):
                stored = feat.detach().cpu().half()
                features[path] = stored
                if cache_root is not None:
                    cache_file = _feature_cache_path(cache_root, root, path)
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(stored, cache_file)

    return features


def _predict_pair_batch(
    model: torch.nn.Module,
    model_kind: str,
    batch_pairs: Sequence[PairEntry],
    feature_cache: dict[Path, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        if model_kind == "geopairnet":
            source_global = torch.stack(
                [feature_cache[pair.source]["global"].float() for pair in batch_pairs]
            ).to(device)
            source_spatial = torch.stack(
                [feature_cache[pair.source]["spatial"].float() for pair in batch_pairs]
            ).to(device)
            target_global = torch.stack(
                [feature_cache[pair.target]["global"].float() for pair in batch_pairs]
            ).to(device)
            target_spatial = torch.stack(
                [feature_cache[pair.target]["spatial"].float() for pair in batch_pairs]
            ).to(device)

            geometry_np = np.stack(
                [build_geometry_features(pair.source.name, pair.target.name) for pair in batch_pairs],
                axis=0,
            )
            geometry = torch.from_numpy(geometry_np).to(device=device, dtype=torch.float32)
            match = torch.zeros(
                len(batch_pairs),
                int(getattr(model, "match_feature_dim", 8)),
                device=device,
                dtype=torch.float32,
            )

            pred = model.forward_from_embeddings(  # type: ignore[attr-defined]
                source_global=source_global,
                source_spatial=source_spatial,
                target_global=target_global,
                target_spatial=target_spatial,
                match_features=match,
                geometry_features=geometry,
            )
            heading = pred["heading_deg"]
            distance = pred["distance"]
            return heading, distance

        source_features = torch.stack([feature_cache[pair.source].float() for pair in batch_pairs]).to(device)
        target_features = torch.stack([feature_cache[pair.target].float() for pair in batch_pairs]).to(device)

        if model_kind == "dual_path":
            pred = model.forward_features(source_features, target_features)  # type: ignore[attr-defined]
            heading = pred["heading"]
            distance = pred["distance"]
            return heading, distance

        fused = siamese_fusion(source_features, target_features)
        if hasattr(model, "regression_head"):
            raw = model.regression_head(fused)
            heading = raw[:, 0]
            distance = torch.clamp(raw[:, 1], min=0.0)
            return heading, distance

        pred = model.head(fused)  # type: ignore[attr-defined]
        return pred["heading_deg"], pred["distance"]


def _format_output_line(angle: float, distance: float, delimiter: DelimiterMode) -> str:
    if delimiter == "space":
        sep = " "
    else:
        sep = ", "
    return f"{angle:.6f}{sep}{distance:.6f}"


def _sanitize_prediction(angle: float, distance: float) -> tuple[float, float]:
    if not math.isfinite(angle) or not math.isfinite(distance):
        raise ValueError(f"Non-finite prediction detected: angle={angle}, distance={distance}")
    wrapped_angle = ((float(angle) + 180.0) % 360.0) - 180.0
    metric_distance = max(1e-6, float(distance))
    return wrapped_angle, metric_distance


def _verify_submission(
    expected_pairs: Sequence[PairEntry],
    output_lines: Sequence[str],
    output_path: Path,
) -> None:
    expected_count = len(expected_pairs)
    predicted_count = len(output_lines)

    if predicted_count != expected_count:
        raise RuntimeError(
            "Submission count mismatch: "
            f"expected={expected_count}, predicted={predicted_count}. "
            "Check pair discovery ordering and output formatting."
        )

    disk_lines = output_path.read_text(encoding="utf-8").splitlines()
    if len(disk_lines) != expected_count:
        raise RuntimeError(
            "Written file line count mismatch: "
            f"expected={expected_count}, on_disk={len(disk_lines)}"
        )

    preview_count = min(5, expected_count)
    print("[verify] pair count matched")
    print("[verify] first resolved pair IDs:")
    for pair in expected_pairs[:preview_count]:
        print(f"  - {pair.pair_id}")
    print("[verify] first output lines:")
    for line in output_lines[:preview_count]:
        print(f"  - {line}")


def _create_submission_zip(result_txt: Path, zip_path: Path) -> None:
    if not result_txt.is_file():
        raise FileNotFoundError(f"Missing result.txt for zip packaging: {result_txt}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(result_txt, arcname="result.txt")


def generate_submission(
    checkpoint: str | None,
    cache_dir: str | None = None,
    pairuav_root: str | None = None,
    output: str = "result.txt",
    pair_order: PairOrderMode = "official",
    delimiter: DelimiterMode = "comma",
    verify: bool = False,
    zip_output: str | None = None,
    dry_run_zip: str | None = None,
    safe_submission_mode: bool = False,
) -> None:
    """Generate result.txt for CodaBench submission with strict parity checks."""
    if safe_submission_mode:
        pair_order = "official"
        delimiter = "comma"
        verify = True

    root = resolve_pairuav_root(pairuav_root)
    pairs, pair_source = _discover_pairs(root, pair_order=pair_order)
    if not pairs:
        raise RuntimeError(f"No test pairs were discovered under {root}.")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Resolved PairUAV root: {root}")
    print(f"Pair order source: {pair_source}")
    print(f"Pair count: {len(pairs)}")

    output_lines: list[str] = []

    if dry_run_zip is not None:
        print("Dry-run mode enabled: writing placeholder predictions only.")
        output_lines = [_format_output_line(0.0, 1.0, delimiter="comma") for _ in pairs]
    else:
        if checkpoint is None:
            raise ValueError("--checkpoint is required unless --dry-run-zip is used")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_kind = _load_model(checkpoint, device)

        unique_images = sorted(
            {pair.source for pair in pairs}.union({pair.target for pair in pairs}),
            key=lambda path: _natural_sort_key(str(path.relative_to(root)).replace("\\", "/")),
        )
        feature_cache = _extract_features(model, model_kind, unique_images, device, root, cache_dir=cache_dir)

        batch_size = 64 if model_kind in {"pose_lite", "geopairnet"} else 32
        print(f"Generating submission -> {output_path}")
        print(f"  Model: {model_kind} | Unique images: {len(unique_images)} | Batch size: {batch_size}")

        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start:start + batch_size]
            heading, distance = _predict_pair_batch(model, model_kind, batch_pairs, feature_cache, device)

            heading_values = heading.detach().cpu().tolist()
            distance_values = distance.detach().cpu().tolist()
            for angle, dist in zip(heading_values, distance_values):
                wrapped_angle, metric_distance = _sanitize_prediction(float(angle), float(dist))
                output_lines.append(_format_output_line(wrapped_angle, metric_distance, delimiter=delimiter))

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for line in output_lines:
            handle.write(line + "\n")

    print(f"Result written to {output_path}")

    if verify:
        _verify_submission(expected_pairs=pairs, output_lines=output_lines, output_path=output_path)

    if zip_output is not None:
        zip_path = Path(zip_output)
        _create_submission_zip(output_path, zip_path)
        print(f"Packaged zip: {zip_path}")

    if dry_run_zip is not None:
        dry_run_zip_path = Path(dry_run_zip)
        _create_submission_zip(output_path, dry_run_zip_path)
        print(f"Dry-run zip ready: {dry_run_zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint path (required unless --dry-run-zip is set)",
    )
    parser.add_argument("--cache", type=str)
    parser.add_argument(
        "--pairuav-root",
        "--university-release",
        "--data",
        dest="pairuav_root",
        type=str,
        default=None,
        help=(
            "Path to the PairUAV data root used for submission discovery; "
            "when omitted the script auto-detects the mounted dataset"
        ),
    )
    parser.add_argument("--output", type=str, default="result.txt")
    parser.add_argument(
        "--pair-order",
        type=str,
        choices=["official", "auto"],
        default="official",
        help="Submission ordering mode. official is required for leaderboard parity.",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        choices=["comma", "space"],
        default="comma",
        help="Result line delimiter. Challenge canonical format is comma.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable strict pair-count and preview verification.",
    )
    parser.add_argument(
        "--zip",
        dest="zip_output",
        type=str,
        default=None,
        help="Optional zip output path. result.txt is stored at zip root.",
    )
    parser.add_argument(
        "--dry-run-zip",
        type=str,
        default=None,
        help="Create a dry-run zip with placeholder result.txt at zip root.",
    )
    parser.add_argument(
        "--safe-submission-mode",
        action="store_true",
        help="Force official order + comma format + verification.",
    )
    args = parser.parse_args()

    if args.checkpoint is None and args.dry_run_zip is None:
        parser.error("--checkpoint is required unless --dry-run-zip is used")

    generate_submission(
        checkpoint=args.checkpoint,
        cache_dir=args.cache,
        pairuav_root=args.pairuav_root,
        output=args.output,
        pair_order=args.pair_order,
        delimiter=args.delimiter,
        verify=bool(args.verify),
        zip_output=args.zip_output,
        dry_run_zip=args.dry_run_zip,
        safe_submission_mode=bool(args.safe_submission_mode),
    )

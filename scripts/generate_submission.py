"""Generate CodaBench submission files for PairUAV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
import torchvision.models as models

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import collect_annotation_json_paths
from data.dataset_pairuav import (
    OfflineMatchFeatureStore,
    build_geometry_features,
    build_pair_key,
    collect_official_test_pairs,
    resolve_test_annotation_dir,
)
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


def _auto_match_root(root: Path, split_tag: str = "test") -> str | None:
    candidates = (
        root / f"{split_tag}_matches_data",
        root / "matches" / f"{split_tag}_matches_data",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate.resolve())
    return None


def _auto_match_index_file(match_root: str | None) -> str | None:
    if not match_root:
        return None

    root = Path(match_root)
    for name in ("index.csv", "match_index.csv", "matches.csv", "pairs.csv"):
        candidate = root / name
        if candidate.is_file():
            return str(candidate.resolve())
    return None


@dataclass(frozen=True)
class SubmissionPair:
    source: Path
    target: Path
    pair_id: str
    source_ref: str
    target_ref: str


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
        key=lambda path: str(path.relative_to(root)).replace("\\", "/").lower(),
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
    by_relative: dict[str, Path],
    by_name: dict[str, list[Path]],
) -> Path | None:
    candidate = Path(ref)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    normalized = ref.strip().lstrip("./").replace("\\", "/")
    direct = root / normalized
    if direct.exists():
        return direct

    for prefix in ("test/", "test_tour/", "train/", "train_tour/"):
        prefixed = root / prefix / normalized
        if prefixed.exists():
            return prefixed

    if normalized in by_relative:
        return by_relative[normalized]

    matches = by_name.get(Path(normalized).name.lower(), [])
    if len(matches) == 1:
        return matches[0]

    return None


def _parse_manifest(
    manifest: Path,
    root: Path,
    by_relative: dict[str, Path],
    by_name: dict[str, list[Path]],
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []

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

            source = _resolve_image_ref(fields[0], root, by_relative, by_name)
            target = _resolve_image_ref(fields[1], root, by_relative, by_name)
            if source is not None and target is not None:
                pairs.append((source, target))

    return pairs


def _group_image_sets(directory: Path) -> dict[str, list[Path]]:
    subdirs = [path for path in directory.iterdir() if path.is_dir()]
    if not subdirs:
        images = _scan_image_files(directory)
        return {"": images} if images else {}

    grouped: dict[str, list[Path]] = {}
    for subdir in sorted(subdirs, key=lambda path: path.name.lower()):
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


def _resolve_official_test_annotation_dir(root: Path) -> Path | None:
    return resolve_test_annotation_dir(root)


def _load_official_submission_pairs(
    root: Path,
    by_relative: dict[str, Path],
    by_name: dict[str, list[Path]],
) -> list[SubmissionPair]:
    test_annotation_dir = _resolve_official_test_annotation_dir(root)
    if test_annotation_dir is None:
        return []

    json_paths = collect_annotation_json_paths(test_annotation_dir)
    output: list[SubmissionPair] = []
    missing_refs: list[str] = []

    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        source_ref = payload.get("image_a") or payload.get("source")
        target_ref = payload.get("image_b") or payload.get("target")
        if not source_ref or not target_ref:
            continue

        source = _resolve_image_ref(str(source_ref), root, by_relative, by_name)
        target = _resolve_image_ref(str(target_ref), root, by_relative, by_name)
        if source is None or target is None:
            missing_refs.append(str(json_path.relative_to(root)).replace("\\", "/"))
            continue

        pair_key = build_pair_key(str(source_ref), str(target_ref))
        pair_id = f"{str(json_path.relative_to(test_annotation_dir)).replace('\\', '/')}::{pair_key}"
        output.append(
            SubmissionPair(
                source=source,
                target=target,
                pair_id=pair_id,
                source_ref=str(source_ref).strip().replace("\\", "/"),
                target_ref=str(target_ref).strip().replace("\\", "/"),
            )
        )

    if missing_refs:
        preview = ", ".join(missing_refs[:5])
        raise FileNotFoundError(
            "Official test annotations exist, but some image references could not be resolved. "
            f"Examples: {preview}"
        )

    return output


def _discover_pairs_fallback(root: Path) -> list[tuple[Path, Path]]:
    by_relative, by_name, _ = _build_image_index(root)

    test_root = root / "test"
    if test_root.is_dir():
        grouped: dict[str, set[str]] = defaultdict(set)
        json_found = False
        for subdir in sorted(test_root.iterdir(), key=lambda path: path.name.lower()):
            if not subdir.is_dir():
                continue
            for json_file in sorted(subdir.glob("*.json")):
                json_found = True
                with json_file.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                image_a = data.get("image_a")
                image_b = data.get("image_b")
                if not image_a or not image_b:
                    continue
                grouped[str(image_a)].add(str(image_b))

        if json_found and grouped:
            ordered_pairs: list[tuple[Path, Path]] = []
            for image_a, image_b_set in sorted(grouped.items()):
                image_a_ref = str(Path(image_a).with_suffix(".webp"))
                for image_b in sorted(image_b_set):
                    source = _resolve_image_ref(image_a_ref, root, by_relative, by_name)
                    target = _resolve_image_ref(image_b, root, by_relative, by_name)
                    if source is not None and target is not None:
                        ordered_pairs.append((source, target))
            if ordered_pairs:
                return ordered_pairs

    for base in (root, root / "test", root / "test_tour"):
        for name in PAIR_MANIFEST_NAMES:
            candidate = base / name
            if candidate.is_file():
                pairs = _parse_manifest(candidate, root, by_relative, by_name)
                if pairs:
                    return pairs

    split_dirs = _find_split_pair_dirs(root)
    if split_dirs is None:
        raise FileNotFoundError(
            f"Could not find test pair data under {root}. "
            "Expected a manifest file or query/gallery split directories."
        )

    query_dir, gallery_dir = split_dirs
    query_groups = _group_image_sets(query_dir)
    gallery_groups = _group_image_sets(gallery_dir)

    pairs: list[tuple[Path, Path]] = []
    common_group_names = [name for name in sorted(query_groups) if name in gallery_groups]

    if common_group_names:
        for group_name in common_group_names:
            query_images = query_groups[group_name]
            gallery_images = gallery_groups[group_name]
            if not query_images or not gallery_images:
                continue
            pair_count = min(len(query_images), len(gallery_images))
            pairs.extend(zip(query_images[:pair_count], gallery_images[:pair_count]))
        if pairs:
            return pairs

    query_flat = _scan_image_files(query_dir)
    gallery_flat = _scan_image_files(gallery_dir)
    pair_count = min(len(query_flat), len(gallery_flat))
    return list(zip(query_flat[:pair_count], gallery_flat[:pair_count]))


def _resolve_submission_pairs(
    root: Path,
    require_official_order: bool,
) -> tuple[list[SubmissionPair], str]:
    by_relative, by_name, _ = _build_image_index(root)

    official_pairs = _load_official_submission_pairs(
        root=root,
        by_relative=by_relative,
        by_name=by_name,
    )
    if official_pairs:
        return official_pairs, "official_test_json"

    if require_official_order:
        raise FileNotFoundError(
            "Official pair-order mode requested, but no official test JSON pair list was found. "
            "Expected processed PairUAV test annotations under <pairuav_root>/test."
        )

    fallback = _discover_pairs_fallback(root)
    fallback_pairs = [
        SubmissionPair(
            source=source,
            target=target,
            pair_id=f"fallback_{idx:08d}",
            source_ref=str(source),
            target_ref=str(target),
        )
        for idx, (source, target) in enumerate(fallback)
    ]
    return fallback_pairs, "fallback_discovery"


def _discover_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs, _ = _resolve_submission_pairs(root, require_official_order=False)
    return [(item.source, item.target) for item in pairs]


def _load_model(checkpoint: str, device: torch.device, use_ema: bool = False) -> tuple[torch.nn.Module, str, str]:
    ckpt = torch.load(checkpoint, map_location=device)
    state_source = "model_state_dict"
    if isinstance(ckpt, dict):
        if use_ema and ckpt.get("ema_state_dict"):
            state_dict = ckpt["ema_state_dict"]
            state_source = "ema_state_dict"
        else:
            state_dict = ckpt.get("model_state_dict", ckpt)
    else:
        state_dict = ckpt
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
        return model.eval(), "geopairnet", state_source

    if dual_path_keys and not pose_lite_keys:
        model = HARPDualPath(frozen=True, use_gate=True)
        model.phase = 2
        model.load_state_dict(state_dict, strict=False)
        return model.eval().to(device), "dual_path", state_source

    if pose_lite_keys:
        model = HARPPoseLite().to(device)
        model.load_state_dict(state_dict, strict=False)
        return model.eval(), "pose_lite", state_source

    try:
        model = HARPPoseLite().to(device)
        model.load_state_dict(state_dict, strict=False)
        return model.eval(), "pose_lite", state_source
    except Exception:
        model = HARPDualPath(frozen=True, use_gate=True)
        model.phase = 2
        model.load_state_dict(state_dict, strict=False)
        return model.eval().to(device), "dual_path", state_source


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
    batch_pairs: Sequence[SubmissionPair],
    feature_cache: dict[Path, Any],
    device: torch.device,
    match_store: OfflineMatchFeatureStore | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
    with torch.inference_mode():
        if model_kind == "geopairnet":
            source_global = torch.stack(
                [feature_cache[item.source]["global"].float() for item in batch_pairs]
            ).to(device)
            source_spatial = torch.stack(
                [feature_cache[item.source]["spatial"].float() for item in batch_pairs]
            ).to(device)
            target_global = torch.stack(
                [feature_cache[item.target]["global"].float() for item in batch_pairs]
            ).to(device)
            target_spatial = torch.stack(
                [feature_cache[item.target]["spatial"].float() for item in batch_pairs]
            ).to(device)

            geometry_np = np.stack(
                [build_geometry_features(item.source_ref, item.target_ref) for item in batch_pairs],
                axis=0,
            )
            geometry = torch.from_numpy(geometry_np).to(device=device, dtype=torch.float32)
            if match_store is not None and match_store.enabled:
                match_np = np.stack(
                    [match_store.get(item.source_ref, item.target_ref) for item in batch_pairs],
                    axis=0,
                )
                match = torch.from_numpy(match_np).to(device=device, dtype=torch.float32)
            else:
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
            raw_payload = {
                "heading_sin": pred.get("heading_sin"),
                "heading_cos": pred.get("heading_cos"),
                "log_distance": pred.get("log_distance"),
            }
            return heading, distance, raw_payload

        source_features = torch.stack([feature_cache[item.source].float() for item in batch_pairs]).to(device)
        target_features = torch.stack([feature_cache[item.target].float() for item in batch_pairs]).to(device)

        if model_kind == "dual_path":
            pred = model.forward_features(source_features, target_features)  # type: ignore[attr-defined]
            heading = pred["heading"]
            distance = pred["distance"]
            return heading, distance, {"heading_raw": heading, "distance_raw": distance}

        fused = siamese_fusion(source_features, target_features)
        if hasattr(model, "regression_head"):
            raw = model.regression_head(fused)
            heading = raw[:, 0]
            distance = torch.clamp(raw[:, 1], min=0.0)
            return heading, distance, {"heading_raw": raw[:, 0], "distance_raw": raw[:, 1]}

        pred = model.head(fused)  # type: ignore[attr-defined]
        return pred["heading_deg"], pred["distance"], {"heading_raw": pred["heading_deg"], "distance_raw": pred["distance"]}


def _format_result_line(angle_value: float, distance_value: float, delimiter: str) -> str:
    if delimiter == "space":
        return f"{angle_value:.6f} {distance_value:.6f}"
    return f"{angle_value:.6f}, {distance_value:.6f}"


def _create_result_zip(result_txt: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(result_txt, arcname="result.txt")


def _assert_submission_decode_units(heading: torch.Tensor, distance: torch.Tensor, context: str) -> None:
    heading_values = heading.detach().float()
    distance_values = distance.detach().float()

    if not torch.isfinite(heading_values).all():
        raise RuntimeError(f"Non-finite heading values encountered during {context}")
    if not torch.isfinite(distance_values).all():
        raise RuntimeError(f"Non-finite distance values encountered during {context}")

    wrapped_heading = ((heading_values + 180.0) % 360.0) - 180.0
    min_heading = float(wrapped_heading.min().item())
    max_heading = float(wrapped_heading.max().item())
    if min_heading < -180.0001 or max_heading > 180.0001:
        raise RuntimeError(
            "Heading appears to be in a non-degree internal space during "
            f"{context}: range=[{min_heading:.4f}, {max_heading:.4f}]"
        )

    min_distance = float(distance_values.min().item())
    if min_distance <= 0.0:
        raise RuntimeError(
            "Distance appears to be in a non-metric internal space during "
            f"{context}: min={min_distance:.6f}"
        )


def _print_submission_decode_debug(
    raw_payload: dict[str, torch.Tensor | None],
    heading: torch.Tensor,
    distance: torch.Tensor,
    max_samples: int,
) -> None:
    if max_samples <= 0:
        return

    count = min(max_samples, int(heading.shape[0]))
    if count <= 0:
        return

    print("Submission decode debug samples:")
    for idx in range(count):
        heading_sin = raw_payload.get("heading_sin")
        heading_cos = raw_payload.get("heading_cos")
        log_distance = raw_payload.get("log_distance")
        heading_raw = raw_payload.get("heading_raw")
        distance_raw = raw_payload.get("distance_raw")

        sin_text = "n/a" if heading_sin is None else f"{float(heading_sin[idx].item()):.4f}"
        cos_text = "n/a" if heading_cos is None else f"{float(heading_cos[idx].item()):.4f}"
        log_dist_text = "n/a" if log_distance is None else f"{float(log_distance[idx].item()):.4f}"
        heading_raw_text = "n/a" if heading_raw is None else f"{float(heading_raw[idx].item()):.4f}"
        distance_raw_text = "n/a" if distance_raw is None else f"{float(distance_raw[idx].item()):.4f}"

        decoded_heading = float((((heading[idx].item() + 180.0) % 360.0) - 180.0))
        decoded_distance = float(distance[idx].item())
        print(
            f"  sample[{idx}] raw(rot_sin={sin_text}, rot_cos={cos_text}, rot_raw={heading_raw_text}, "
            f"dist_log={log_dist_text}, dist_raw={distance_raw_text}) "
            f"-> heading_deg={decoded_heading:.4f}, distance_m={decoded_distance:.4f}"
        )


def _log_submission_layout(root: Path, pairs: Sequence[SubmissionPair]) -> None:
    train_annotation_dir = None
    if (root / "train").is_dir() and collect_annotation_json_paths(root / "train"):
        train_annotation_dir = root / "train"

    test_annotation_dir = _resolve_official_test_annotation_dir(root)
    tour_annotation_dir = None
    for candidate in (root / "test_tour", root / "tour"):
        if candidate.is_dir() and collect_annotation_json_paths(candidate):
            tour_annotation_dir = candidate
            break

    train_image_dir = None
    for candidate in (root / "train" / "drone", root / "train_tour", root / "train"):
        if candidate.is_dir():
            train_image_dir = candidate
            break

    test_image_dir = None
    for candidate in (root / "test" / "drone", root / "test", root / "test_tour"):
        if candidate.is_dir():
            test_image_dir = candidate
            break

    tour_image_dir = None
    for candidate in (root / "test_tour" / "drone", root / "test_tour", root / "tour"):
        if candidate.is_dir():
            tour_image_dir = candidate
            break

    print("Resolved PairUAV layout:")
    print(f"  data_root={root}")
    print(f"  train_image_dir={train_image_dir}")
    print(f"  train_annotation_dir={train_annotation_dir}")
    print(f"  test_image_dir={test_image_dir}")
    print(f"  test_annotation_dir={test_annotation_dir}")
    print(f"  tour_annotation_dir={tour_annotation_dir}")
    print(f"  tour_image_dir={tour_image_dir}")
    print(f"  test_pair_count={len(pairs)}")

    preview_count = min(5, len(pairs))
    if preview_count > 0:
        print("  first 5 official test pairs:")
        for item in pairs[:preview_count]:
            print(f"    {item.pair_id} -> {item.source_ref} || {item.target_ref}")


def _resolve_submission_match_store(
    model: torch.nn.Module,
    model_kind: str,
    pairs: Sequence[SubmissionPair],
    match_root: str | None,
    match_index_file: str | None,
    match_missing_policy: str,
) -> OfflineMatchFeatureStore | None:
    if model_kind != "geopairnet":
        return None

    if bool(getattr(model, "no_match_features", False)):
        print("Match features are disabled in the checkpoint model config (no_match_features=True).")
        return None

    policy = match_missing_policy.lower().strip()
    if policy not in {"disable", "error"}:
        policy = "disable"

    if not match_root:
        message = (
            "Match features expected but no --match-root was provided "
            "(expected test_matches_data/ for official pipeline)."
        )
        if policy == "error":
            raise FileNotFoundError(message)
        print(f"WARNING: {message} Entering no-match-features fallback mode.")
        return None

    root_path = Path(match_root).expanduser().resolve()
    if not root_path.is_dir():
        message = f"Match root does not exist: {root_path}"
        if policy == "error":
            raise FileNotFoundError(message)
        print(f"WARNING: {message}. Entering no-match-features fallback mode.")
        return None

    index_path = str(Path(match_index_file).expanduser().resolve()) if match_index_file else None
    store = OfflineMatchFeatureStore(
        match_root=str(root_path),
        index_file=index_path,
        feature_dim=int(getattr(model, "match_feature_dim", 8)),
    )

    sample_count = min(5, len(pairs))
    missing = 0
    reverse_failures = 0
    reverse_checked = 0
    if sample_count > 0:
        print("Match alignment check (official test samples):")
    for item in pairs[:sample_count]:
        probe = store.probe(item.source_ref, item.target_ref)
        status = "FOUND" if probe.found else "MISSING"
        reverse_text = "reverse" if probe.used_reverse else "direct"
        print(f"  {item.pair_id}: {status} ({reverse_text})")
        if not probe.found:
            missing += 1
            continue

        reverse_probe = store.probe(item.target_ref, item.source_ref)
        if reverse_probe.found and (probe.used_reverse ^ reverse_probe.used_reverse):
            forward_feat = store.get(item.source_ref, item.target_ref)
            reverse_feat = store.get(item.target_ref, item.source_ref)
            reverse_checked += 1
            if abs(float(forward_feat[2] + reverse_feat[2])) > 2e-2:
                reverse_failures += 1
            if abs(float(forward_feat[3] + reverse_feat[3])) > 2e-2:
                reverse_failures += 1

    if reverse_checked > 0:
        print(
            f"  reverse-pair consistency checks: {reverse_checked} samples, "
            f"failures={reverse_failures}"
        )

    if missing > 0 or reverse_failures > 0:
        message = f"Match alignment failed (missing={missing}, reverse_failures={reverse_failures})"
        if policy == "error":
            raise RuntimeError(message)
        print(f"WARNING: {message}. Entering no-match-features fallback mode.")
        return None

    print(f"Match features enabled for submission: root={root_path}, index={index_path}")
    return store


def _verify_submission_output(
    pairs: Sequence[SubmissionPair],
    first_output_lines: Sequence[str],
    predicted_count: int,
    expected_count: int,
) -> None:
    print("Verification summary:")
    print(f"  Expected pairs: {expected_count}")
    print(f"  Predicted lines: {predicted_count}")

    if predicted_count != expected_count:
        raise RuntimeError(
            "Submission line count mismatch: "
            f"expected {expected_count}, got {predicted_count}."
        )

    unique_pair_ids = len({item.pair_id for item in pairs})
    if unique_pair_ids != len(pairs):
        raise RuntimeError(
            "Pair-order ambiguity detected: pair ids are not unique "
            f"(unique={unique_pair_ids}, total={len(pairs)})."
        )

    preview_count = min(10, len(pairs))
    print(f"  First {preview_count} official pair ids:")
    for item in pairs[:preview_count]:
        official_pair_id = item.pair_id.split("::", 1)[0]
        print(f"    {official_pair_id} -> {item.source_ref} || {item.target_ref}")

    print(f"  First {preview_count} output lines:")
    for line in first_output_lines[:preview_count]:
        print(f"    {line}")


def generate_submission(
    checkpoint: str,
    cache_dir: str | None = None,
    pairuav_root: str | None = None,
    output: str = "result.txt",
    split: str = "query",
    pair_order: str = "official",
    delimiter: str = "comma",
    verify: bool = False,
    dry_run_zip: str | None = None,
    match_root: str | None = None,
    match_index_file: str | None = None,
    match_missing_policy: str = "disable",
    decode_debug_samples: int = 0,
    safe_submission_mode: bool = False,
    use_ema: bool = False,
) -> None:
    """Generate result.txt for CodaBench submission."""
    _ = split
    safe_overrides: list[str] = []
    if safe_submission_mode:
        if pair_order != "official":
            pair_order = "official"
            safe_overrides.append("pair_order=official")
        if not verify:
            verify = True
            safe_overrides.append("verify=True")
        if use_ema:
            use_ema = False
            safe_overrides.append("use_ema=False")
        if match_missing_policy != "disable":
            match_missing_policy = "disable"
            safe_overrides.append("match_missing_policy=disable")

    if safe_submission_mode:
        print("safe_submission_mode enabled")
        if safe_overrides:
            print("safe_submission_mode overrides:")
            for item in safe_overrides:
                print(f"  - {item}")
        else:
            print("safe_submission_mode overrides: none")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_pairuav_root(pairuav_root)
    require_official_order = pair_order == "official"

    model, model_kind, state_source = _load_model(checkpoint, device, use_ema=use_ema)
    pairs, pair_source = _resolve_submission_pairs(
        root=root,
        require_official_order=require_official_order,
    )
    if not pairs:
        raise RuntimeError(f"No test pairs were discovered under {root}.")

    _log_submission_layout(root, pairs)

    resolved_match_root = match_root
    resolved_match_index_file = match_index_file
    if resolved_match_root is None:
        resolved_match_root = _auto_match_root(root, split_tag="test")
        if resolved_match_root is not None:
            print(f"Detected test match root: {resolved_match_root}")
    if resolved_match_index_file is None:
        resolved_match_index_file = _auto_match_index_file(resolved_match_root)
        if resolved_match_index_file is not None:
            print(f"Detected test match index: {resolved_match_index_file}")

    match_store = _resolve_submission_match_store(
        model=model,
        model_kind=model_kind,
        pairs=pairs,
        match_root=resolved_match_root,
        match_index_file=resolved_match_index_file,
        match_missing_policy=match_missing_policy,
    )

    pair_paths = [(item.source, item.target) for item in pairs]

    unique_images = sorted({path for pair in pair_paths for path in pair}, key=lambda path: str(path).lower())
    feature_cache = _extract_features(model, model_kind, unique_images, device, root, cache_dir=cache_dir)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = 64 if model_kind in {"pose_lite", "geopairnet"} else 32
    print(f"Generating submission -> {output_path}")
    print(
        f"  Pairs: {len(pairs)} | Unique images: {len(unique_images)} | "
        f"Model: {model_kind} | Pair order source: {pair_source} | checkpoint_state={state_source}"
    )

    written_line_count = 0
    first_output_lines: list[str] = []
    printed_decode_debug = False

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for start in range(0, len(pairs), batch_size):
            batch_items = pairs[start:start + batch_size]
            heading, distance, raw_payload = _predict_pair_batch(
                model,
                model_kind,
                batch_items,
                feature_cache,
                device,
                match_store=match_store,
            )

            _assert_submission_decode_units(heading, distance, context=f"submission batch start={start}")
            if decode_debug_samples > 0 and not printed_decode_debug:
                _print_submission_decode_debug(raw_payload, heading, distance, max_samples=decode_debug_samples)
                printed_decode_debug = True

            heading_values = heading.detach().cpu().tolist()
            distance_values = distance.detach().cpu().tolist()
            for angle, dist in zip(heading_values, distance_values):
                angle_value = ((float(angle) + 180.0) % 360.0) - 180.0
                distance_value = max(0.0, float(dist))
                line = _format_result_line(
                    angle_value=angle_value,
                    distance_value=distance_value,
                    delimiter=delimiter,
                )
                handle.write(f"{line}\n")
                written_line_count += 1
                if len(first_output_lines) < 10:
                    first_output_lines.append(line)

    expected_count = len(pairs)
    if written_line_count != expected_count:
        raise RuntimeError(
            "Generated line count mismatch: "
            f"expected {expected_count}, wrote {written_line_count}."
        )

    if verify:
        _verify_submission_output(
            pairs=pairs,
            first_output_lines=first_output_lines,
            predicted_count=written_line_count,
            expected_count=expected_count,
        )

    if dry_run_zip:
        zip_path = Path(dry_run_zip)
        _create_result_zip(result_txt=output_path, zip_path=zip_path)
        print(f"Dry-run zip created: {zip_path} (result.txt at archive root)")

    print(f"Result written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
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
        choices=["official", "auto"],
        default="official",
        help="Use strict official processed test JSON order or fallback auto discovery.",
    )
    parser.add_argument(
        "--delimiter",
        choices=["comma", "space"],
        default="comma",
        help="Output delimiter format for result lines.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable strict verification output (counts + first 10 pair ids + first 10 lines).",
    )
    parser.add_argument(
        "--dry-run-zip",
        type=str,
        default=None,
        help="Optional zip output path to package result.txt at archive root.",
    )
    parser.add_argument(
        "--match-root",
        type=str,
        default=None,
        help="Optional root directory containing processed SuperGlue match summaries for test pairs.",
    )
    parser.add_argument(
        "--match-index-file",
        type=str,
        default=None,
        help="Optional CSV index mapping (source,target)->match summary file.",
    )
    parser.add_argument(
        "--match-missing-policy",
        choices=["disable", "error"],
        default="disable",
        help="Behavior when match features are expected but missing/misaligned.",
    )
    parser.add_argument(
        "--decode-debug-samples",
        type=int,
        default=0,
        help="Print raw/decode details for N submission samples.",
    )
    parser.add_argument(
        "--safe-submission-mode",
        action="store_true",
        help="Force strict official order and verification safeguards for upload-ready submissions.",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use ema_state_dict from checkpoint if available.",
    )
    args = parser.parse_args()

    generate_submission(
        checkpoint=args.checkpoint,
        cache_dir=args.cache,
        pairuav_root=args.pairuav_root,
        output=args.output,
        pair_order=args.pair_order,
        delimiter=args.delimiter,
        verify=args.verify,
        dry_run_zip=args.dry_run_zip,
        match_root=args.match_root,
        match_index_file=args.match_index_file,
        match_missing_policy=args.match_missing_policy,
        decode_debug_samples=args.decode_debug_samples,
        safe_submission_mode=args.safe_submission_mode,
        use_ema=args.use_ema,
    )

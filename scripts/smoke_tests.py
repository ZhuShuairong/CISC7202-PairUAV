#!/usr/bin/env python3
"""Lightweight smoke tests for PairUAV correctness checks."""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.geopairnet import GeoPairNet
from scripts.generate_submission import _discover_pairs, generate_submission
from training.losses import PairUAVLoss
from training.train_pairuav import FrozenFeatureCache


def _build_small_geopairnet() -> GeoPairNet:
	model = GeoPairNet(
		backbone_name="resnet50",
		pretrained=False,
		global_dim=128,
		spatial_dim=32,
		fused_dim=128,
		rotation_hidden_dim=64,
		distance_hidden_dim=64,
		distance_bins=8,
		log_distance_min=0.0,
		log_distance_max=5.2,
		match_feature_dim=8,
		geometry_feature_dim=6,
		dropout=0.0,
		use_uncertainty=False,
	)
	return model


def _make_tiny_image(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	pixels = np.full((16, 16, 3), 127, dtype=np.uint8)
	Image.fromarray(pixels).save(path)


def _dummy_batch(batch_size: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	source = torch.randn(batch_size, 3, 224, 224)
	target = torch.randn(batch_size, 3, 224, 224)
	match = torch.randn(batch_size, 8)
	geometry = torch.randn(batch_size, 6)
	return source, target, match, geometry


def test_one_batch_forward() -> None:
	model = _build_small_geopairnet().eval()
	source, target, match, geometry = _dummy_batch(batch_size=2)
	with torch.no_grad():
		pred = model(source, target, match_features=match, geometry_features=geometry)

	required_keys = {"heading_deg", "distance", "log_distance", "heading_sin", "heading_cos"}
	missing = sorted(required_keys - set(pred.keys()))
	if missing:
		raise RuntimeError(f"Missing required prediction keys: {missing}")

	if not torch.isfinite(pred["heading_deg"]).all() or not torch.isfinite(pred["distance"]).all():
		raise RuntimeError("Forward output contains non-finite heading/distance")

	if torch.any(pred["distance"] < 0):
		raise RuntimeError("Forward output produced negative distance")


def test_decode_units() -> None:
	model = _build_small_geopairnet().eval()
	source, target, match, geometry = _dummy_batch(batch_size=3)
	with torch.no_grad():
		pred = model(source, target, match_features=match, geometry_features=geometry)

	heading = pred["heading_deg"]
	distance = pred["distance"]
	log_distance = pred["log_distance"]

	exported_heading = ((heading + 180.0) % 360.0) - 180.0
	exported_distance = torch.clamp(distance, min=1e-6)

	if torch.any(exported_heading < -180.0001) or torch.any(exported_heading > 180.0001):
		raise RuntimeError("Decoded heading is outside [-180, 180] after wrap")
	if torch.any(exported_distance <= 0.0):
		raise RuntimeError("Decoded distance is not positive in export space")

	recon_error = float((torch.exp(log_distance) - distance).abs().max().item())
	if recon_error > 1e-3:
		raise RuntimeError(
			"Distance decode mismatch: expected distance ~= exp(log_distance), "
			f"max_abs_error={recon_error:.6f}"
		)


def test_cached_vs_uncached() -> None:
	model = _build_small_geopairnet().eval()
	source, target, match, geometry = _dummy_batch(batch_size=4)

	with torch.no_grad():
		uncached = model(source, target, match_features=match, geometry_features=geometry)

	cache = FrozenFeatureCache(max_items=16)
	source_ids = [f"s_{idx}" for idx in range(source.shape[0])]
	target_ids = [f"t_{idx}" for idx in range(target.shape[0])]

	with torch.no_grad():
		source_global, source_spatial = cache.encode(
			model,
			source,
			source_ids,
			device=torch.device("cpu"),
			channels_last=False,
		)
		target_global, target_spatial = cache.encode(
			model,
			target,
			target_ids,
			device=torch.device("cpu"),
			channels_last=False,
		)
		cached = model.forward_from_embeddings(
			source_global=source_global,
			source_spatial=source_spatial,
			target_global=target_global,
			target_spatial=target_spatial,
			match_features=match,
			geometry_features=geometry,
		)

	max_heading_diff = float((uncached["heading_deg"] - cached["heading_deg"]).abs().max().item())
	max_distance_diff = float((uncached["distance"] - cached["distance"]).abs().max().item())
	tolerance = 5e-2
	if max(max_heading_diff, max_distance_diff) > tolerance:
		raise RuntimeError(
			"cached-vs-uncached mismatch exceeded tolerance; "
			f"heading_diff={max_heading_diff:.6f}, distance_diff={max_distance_diff:.6f}, "
			f"tolerance={tolerance:.6f}"
		)


def test_overfit_16_samples() -> None:
	model = _build_small_geopairnet().train()
	criterion = PairUAVLoss(
		log_distance_min=0.0,
		log_distance_max=5.2,
		num_bins=8,
		min_distance=1.0,
		smooth_l1_beta=0.05,
	)
	optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

	sample_count = 16
	source_global = torch.randn(sample_count, model.global_dim)
	target_global = torch.randn(sample_count, model.global_dim)
	source_spatial = torch.randn(sample_count, model.spatial_dim, 7, 7)
	target_spatial = torch.randn(sample_count, model.spatial_dim, 7, 7)
	match = torch.randn(sample_count, 8)
	geometry = torch.randn(sample_count, 6)
	target_heading = torch.zeros(sample_count)
	target_distance = torch.full((sample_count,), 12.0)

	with torch.no_grad():
		initial_pred = model.forward_from_embeddings(
			source_global=source_global,
			source_spatial=source_spatial,
			target_global=target_global,
			target_spatial=target_spatial,
			match_features=match,
			geometry_features=geometry,
		)
		initial_loss = float(
			criterion(
				prediction=initial_pred,
				target={"heading": target_heading, "distance": target_distance},
				progress=0.0,
				stage_name="A",
			)["total"].item()
		)

	steps = 60
	for step in range(steps):
		prediction = model.forward_from_embeddings(
			source_global=source_global,
			source_spatial=source_spatial,
			target_global=target_global,
			target_spatial=target_spatial,
			match_features=match,
			geometry_features=geometry,
		)
		loss = criterion(
			prediction=prediction,
			target={"heading": target_heading, "distance": target_distance},
			progress=float(step + 1) / float(steps),
			stage_name="A",
		)["total"]

		if not torch.isfinite(loss):
			raise RuntimeError(f"Loss became non-finite during overfit test at step={step}")

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

	model.eval()
	with torch.no_grad():
		final_pred = model.forward_from_embeddings(
			source_global=source_global,
			source_spatial=source_spatial,
			target_global=target_global,
			target_spatial=target_spatial,
			match_features=match,
			geometry_features=geometry,
		)
		final_loss = float(
			criterion(
				prediction=final_pred,
				target={"heading": target_heading, "distance": target_distance},
				progress=1.0,
				stage_name="A",
			)["total"].item()
		)

	if final_loss >= initial_loss * 0.7:
		raise RuntimeError(
			"Overfit test did not reduce loss enough on 16 samples; "
			f"initial_loss={initial_loss:.6f}, final_loss={final_loss:.6f}"
		)


def test_submission_order() -> None:
	with tempfile.TemporaryDirectory(prefix="pairuav_smoke_") as tmp_dir:
		root = Path(tmp_dir)
		test_dir = root / "test"

		_make_tiny_image(root / "test" / "query_drone" / "image-1.webp")
		_make_tiny_image(root / "test" / "query_drone" / "image-2.webp")
		_make_tiny_image(root / "test" / "query_drone" / "image-3.webp")
		_make_tiny_image(root / "test" / "gallery_drone" / "target-1.webp")
		_make_tiny_image(root / "test" / "gallery_drone" / "target-2.webp")
		_make_tiny_image(root / "test" / "gallery_drone" / "target-3.webp")

		(test_dir / "10").mkdir(parents=True, exist_ok=True)
		(test_dir / "2").mkdir(parents=True, exist_ok=True)

		(test_dir / "10" / "pair_10.json").write_text(
			'{"image_a": "query_drone/image-3.webp", "image_b": "gallery_drone/target-3.webp"}',
			encoding="utf-8",
		)
		(test_dir / "2" / "pair_2.json").write_text(
			'{"image_a": "query_drone/image-1.webp", "image_b": "gallery_drone/target-1.webp"}',
			encoding="utf-8",
		)
		(test_dir / "10" / "pair_1.json").write_text(
			'{"image_a": "query_drone/image-2.webp", "image_b": "gallery_drone/target-2.webp"}',
			encoding="utf-8",
		)

		pairs, source = _discover_pairs(root, pair_order="official")
		if source != "official:test_json":
			raise RuntimeError(f"Expected official json pair source, got {source}")
		if len(pairs) != 3:
			raise RuntimeError(f"Expected 3 discovered pairs, got {len(pairs)}")

		first_ids = [entry.pair_id for entry in pairs[:3]]
		if not first_ids[0].endswith("query_drone/image-1.webp||gallery_drone/target-1.webp"):
			raise RuntimeError("Official pair order mismatch for first pair")

		result_txt = root / "result.txt"
		result_zip = root / "result.zip"
		generate_submission(
			checkpoint=None,
			pairuav_root=str(root),
			output=str(result_txt),
			dry_run_zip=str(result_zip),
			safe_submission_mode=True,
		)

		if not result_txt.is_file():
			raise RuntimeError("Dry-run submission did not create result.txt")
		lines = result_txt.read_text(encoding="utf-8").splitlines()
		if len(lines) != 3:
			raise RuntimeError(f"Expected 3 lines in dry-run result.txt, got {len(lines)}")

		with zipfile.ZipFile(result_zip, "r") as archive:
			names = archive.namelist()
			if names != ["result.txt"]:
				raise RuntimeError(
					"Dry-run zip must contain only result.txt at archive root; "
					f"found={names}"
				)


TESTS = {
	"test_one_batch_forward": test_one_batch_forward,
	"test_overfit_16_samples": test_overfit_16_samples,
	"test_submission_order": test_submission_order,
	"test_decode_units": test_decode_units,
	"test_cached_vs_uncached": test_cached_vs_uncached,
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="PairUAV smoke tests")
	parser.add_argument(
		"--test",
		type=str,
		default="all",
		choices=["all", *TESTS.keys()],
		help="Single test to run or all",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	selected = list(TESTS.keys()) if args.test == "all" else [args.test]
	for name in selected:
		print(f"[Smoke] RUN {name}")
		try:
			TESTS[name]()
		except Exception as exc:
			raise RuntimeError(f"[Smoke] FAIL {name}: {exc}") from exc
		print(f"[Smoke] PASS {name}")

	print("[Smoke] All selected tests passed")


if __name__ == "__main__":
	main()

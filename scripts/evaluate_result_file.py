#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.metrics import evaluate_result_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PairUAV result.txt against truth.txt")
    parser.add_argument("--result", type=str, required=True, help="Path to result.txt-style prediction file")
    parser.add_argument("--truth", type=str, required=True, help="Path to truth.txt-style target file")
    parser.add_argument("--min-distance-denominator", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_result_files(
        result_path=args.result,
        truth_path=args.truth,
        min_distance_denominator=float(args.min_distance_denominator),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

"""Convert a parquet file to JSONL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _to_jsonable(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if np is not None:
        if isinstance(value, np.ndarray):
            return [_to_jsonable(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return _to_jsonable(value.item())
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if converted is not value:
            return _to_jsonable(converted)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a parquet file to JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the input parquet file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to the output JSONL file. Defaults to the input path with a .jsonl suffix.",
    )
    return parser.parse_args()


def parquet_to_jsonl(input_path: Path, output_path: Path) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet file not found: {input_path}")
    if input_path.is_dir():
        raise IsADirectoryError(f"Input path must be a parquet file, got directory: {input_path}")

    df = pd.read_parquet(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")

    return len(df)


def main() -> int:
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_path.with_suffix(".jsonl")
    )

    count = parquet_to_jsonl(input_path, output_path)
    print(f"Converted {count} records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Merge independent benchmark summary JSON files into one JSONL report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_summary_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Summary file is empty: {path}")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object.")
            rows.append(row)
        return rows

    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        return payload
    raise TypeError(f"{path} must contain JSON objects, a JSON object list, or JSONL.")


def merge_summaries(input_files: list[Path], output_file: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for input_file in input_files:
        rows.extend(_load_summary_rows(input_file))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_file.with_suffix(output_file.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_output.replace(output_file)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=Path, nargs="+", required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = merge_summaries(args.input_files, args.output_file)
    print(json.dumps({"merged": len(rows), "output": str(args.output_file)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

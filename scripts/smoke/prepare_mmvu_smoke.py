#!/usr/bin/env python3
"""Create a tiny local-video MMVU JSONL file for the LoRA smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import unquote, urlparse


def _video_relative_path(raw_video: str) -> Path:
    parsed_path = unquote(urlparse(raw_video).path)
    marker = "/videos/"
    if marker not in parsed_path:
        raise ValueError(f"MMVU video URL does not contain {marker!r}: {raw_video}")
    return Path("videos") / parsed_path.split(marker, maxsplit=1)[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmvu_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    args = parser.parse_args()

    if args.count < 1:
        raise ValueError("--count must be positive.")

    validation_path = args.mmvu_dir / "validation.json"
    with validation_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"{validation_path} must contain a JSON list.")

    # Prefer small, unique local videos so decoding remains a true video-path
    # test while keeping the one-GPU smoke run short.
    candidates: list[tuple[int, Path, dict]] = []
    seen_paths: set[Path] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("video"), str):
            continue
        try:
            relative_path = _video_relative_path(record["video"])
        except ValueError:
            continue
        local_path = (args.mmvu_dir / relative_path).resolve()
        if local_path in seen_paths or not local_path.is_file():
            continue
        seen_paths.add(local_path)
        candidates.append((local_path.stat().st_size, local_path, record))

    candidates.sort(key=lambda item: (item[0], str(item[1])))
    selected = candidates[: args.count]
    if len(selected) != args.count:
        raise RuntimeError(
            f"Requested {args.count} unique MMVU videos, but only found {len(selected)} local files."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for _, local_path, record in selected:
            item = {
                "problem": "Generate exactly one reasoning question based on this video.",
                "answer": str(record.get("answer", "unused")),
                "videos": [str(local_path)],
                "data_type": "video",
                "problem_type": "question generation",
                "problem_id": str(record.get("id", local_path.stem)),
                "source_question": str(record.get("question", "")),
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} MMVU smoke samples to {args.output}")
    for size, local_path, _ in selected:
        print(f"  {local_path} ({size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recover a few historical Questioner outputs as evaluate smoke inputs.

The July smoke Questioner emitted its type in the opening tag name rather than
inside ``<type>``.  This helper is test-fixture-only: it recovers those questions
without weakening the production Questioner parser.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_QUESTION_PATTERN = re.compile(r"<question>\s*(.*?)\s*</question>", re.IGNORECASE | re.DOTALL)
_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_LEGACY_TYPE_PATTERN = re.compile(
    r"<(multiple choice|numerical|regression)>\s*(.*?)\s*</\1>",
    re.IGNORECASE | re.DOTALL,
)


def _recover(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = str(row.get("questioner_response", ""))
    question_matches = _QUESTION_PATTERN.findall(raw)
    answer_matches = _ANSWER_PATTERN.findall(raw)
    type_matches = _LEGACY_TYPE_PATTERN.findall(raw)
    if question_matches:
        question = question_matches[-1].strip()
    elif type_matches:
        question = type_matches[-1][1].strip()
    else:
        return None
    if not answer_matches or not type_matches or not question:
        return None

    result = dict(row)
    result.update(
        {
            "question_type": type_matches[-1][0].strip().lower(),
            "question": question,
            "answer": answer_matches[-1].strip(),
            "score": 0,
        }
    )
    result.pop("error", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    recovered = [item for row in rows if (item := _recover(row)) is not None][: args.limit]
    if len(recovered) < args.limit:
        raise RuntimeError(f"Recovered only {len(recovered)} rows; requested {args.limit}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(recovered, handle, ensure_ascii=False, indent=2)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "rows": len(recovered),
                "unique_videos": len({row["video_pt_path"] for row in recovered}),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

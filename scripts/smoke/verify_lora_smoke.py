#!/usr/bin/env python3
"""Verify that both smoke stages trained and the second changed the first adapter."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def _flatten_numbers(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    flattened: list[tuple[str, float]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.extend(_flatten_numbers(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        flattened.append((prefix, float(value)))
    return flattened


def _positive_grad_norms(experiment_log: Path) -> list[float]:
    values: list[float] = []
    with experiment_log.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for key, value in _flatten_numbers(row):
                if key.endswith("grad_norm") and math.isfinite(value) and value > 0:
                    values.append(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage_a", type=Path, required=True)
    parser.add_argument("--stage_b", type=Path, required=True)
    parser.add_argument("--stage_b_console_log", type=Path, required=True)
    args = parser.parse_args()

    adapter_a = args.stage_a / "global_step_1" / "actor" / "lora_adapter"
    adapter_b = args.stage_b / "global_step_1" / "actor" / "lora_adapter"
    for adapter in (adapter_a, adapter_b):
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            path = adapter / filename
            if not path.is_file():
                raise FileNotFoundError(path)

    state_a = load_file(str(adapter_a / "adapter_model.safetensors"), device="cpu")
    state_b = load_file(str(adapter_b / "adapter_model.safetensors"), device="cpu")
    if state_a.keys() != state_b.keys():
        raise RuntimeError("Stage A and B adapter tensor keys differ.")

    changed = 0
    max_abs_diff = 0.0
    for key in state_a:
        if state_a[key].shape != state_b[key].shape:
            raise RuntimeError(f"Adapter tensor shape differs for {key}.")
        diff = torch.max(torch.abs(state_a[key].float() - state_b[key].float())).item()
        max_abs_diff = max(max_abs_diff, diff)
        changed += int(diff > 0.0)

    lora_b_keys = [key for key in state_a if "lora_B" in key]
    stage_a_lora_b_max = max(
        (torch.max(torch.abs(state_a[key].float())).item() for key in lora_b_keys),
        default=0.0,
    )
    stage_a_grad_norms = _positive_grad_norms(args.stage_a / "experiment_log.jsonl")
    stage_b_grad_norms = _positive_grad_norms(args.stage_b / "experiment_log.jsonl")
    stage_b_console = args.stage_b_console_log.read_text(encoding="utf-8", errors="replace")
    loaded_marker = "Loading trainable LoRA:" in stage_b_console

    report = {
        "adapter_a": str(adapter_a),
        "adapter_b": str(adapter_b),
        "tensor_count": len(state_a),
        "changed_tensor_count": changed,
        "max_abs_diff": max_abs_diff,
        "stage_a_lora_b_max_abs": stage_a_lora_b_max,
        "stage_a_positive_grad_norms": stage_a_grad_norms,
        "stage_b_positive_grad_norms": stage_b_grad_norms,
        "stage_b_loaded_adapter_marker": loaded_marker,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not lora_b_keys or stage_a_lora_b_max <= 0.0:
        raise RuntimeError("Stage A has no non-zero LoRA-B tensor; an update was not demonstrated.")
    if not stage_a_grad_norms:
        raise RuntimeError("Stage A did not log a positive finite actor grad_norm.")
    if not loaded_marker:
        raise RuntimeError("Stage B did not log the trainable LoRA loading marker.")
    if not stage_b_grad_norms:
        raise RuntimeError("Stage B did not log a positive finite actor grad_norm.")
    if changed == 0 or max_abs_diff <= 0.0:
        raise RuntimeError("Stage B adapter is identical to Stage A.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU smoke test for EvoVid Solver reward and target-segment plumbing."""

from __future__ import annotations

import math

import numpy as np

from examples.reward_function.evo_vid_solver_reward import compute_score
from verl.protocol import DataProto
from verl.workers.reward.function import _build_reward_input


def _assert_close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
        raise AssertionError(f"expected {expected}, got {actual}")


def main() -> None:
    data = DataProto.from_dict(
        non_tensors={
            "ground_truth": np.array(["B"], dtype=object),
            "problem_type": np.array(["multiple choice"], dtype=object),
            "target_segment": np.array([[2.0, 6.0]], dtype=object),
        }
    )
    reward_input = _build_reward_input(
        data,
        "Reasoning. \\boxed{B} <segment>4s-8s</segment>",
        response_length=12,
        index=0,
    )
    if list(reward_input["target_segment"]) != [2.0, 6.0]:
        raise AssertionError("target_segment was not copied into RewardInput")

    cases = [
        reward_input,
        {
            **reward_input,
            "response": "Reasoning. \\boxed{C} <segment>2s-6s</segment>",
        },
        {
            **reward_input,
            "response": "Reasoning. \\boxed{B}",
        },
        {
            **reward_input,
            "response": "Reasoning. B <segment>2s-6s</segment>",
        },
        {
            **reward_input,
            "response": "Reasoning. \\boxed{B} <segment>-1s-6s</segment>",
        },
    ]
    scores = compute_score(cases)

    # Pred [4, 8], target [2, 6]: intersection=2, union=6, IoU=1/3.
    _assert_close(scores[0]["accuracy"], 1.0)
    _assert_close(scores[0]["format"], 1.0)
    _assert_close(scores[0]["iou"], 1.0 / 3.0)
    _assert_close(scores[0]["temporal"], 1.0 / 3.0)
    _assert_close(scores[0]["overall"], 1.1)

    # IoU is logged for a wrong answer, but the temporal bonus is gated by accuracy.
    _assert_close(scores[1]["accuracy"], 0.0)
    _assert_close(scores[1]["format"], 1.0)
    _assert_close(scores[1]["iou"], 1.0)
    _assert_close(scores[1]["temporal"], 0.0)
    _assert_close(scores[1]["overall"], 0.1)

    # Missing segment: correct answer still gets 0.9, but no format/IoU bonus.
    _assert_close(scores[2]["overall"], 0.9)
    _assert_close(scores[2]["format"], 0.0)
    _assert_close(scores[2]["iou"], 0.0)

    # Missing boxed answer means zero accuracy even when a segment exists.
    _assert_close(scores[3]["accuracy"], 0.0)
    _assert_close(scores[3]["overall"], 0.0)

    # Invalid negative timestamps cannot earn format or temporal reward.
    _assert_close(scores[4]["format"], 0.0)
    _assert_close(scores[4]["iou"], 0.0)
    _assert_close(scores[4]["overall"], 0.9)

    print("solver IoU reward smoke: PASS")


if __name__ == "__main__":
    main()

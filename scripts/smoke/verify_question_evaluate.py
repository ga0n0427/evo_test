#!/usr/bin/env python3
"""CPU contract checks for EvoVid full-video question evaluation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from jinja2 import Template

from question_evaluate.evaluate import (
    _build_video_input,
    _common_result_fields,
    _extract_candidate_segment,
    score_solver_responses,
)
from question_evaluate.upload import curate_rows


class _FakeProcessor:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt is True
        assert tokenize is False
        content = messages[0]["content"]
        assert content[0] == {"type": "video"}
        return f"<video>{content[1]['text']}"


class _FakeTokenizer:
    def encode(self, prompt, *, add_special_tokens):
        assert prompt.count("<video>") == 1
        assert add_special_tokens is False
        return [101, 102, 103]


def main() -> None:
    response_texts = [
        "reasoning \\boxed{A} <segment>1.0s-3.0s</segment>",
        "reasoning \\boxed{A} <segment>1s–3s</segment>",
        "reasoning \\boxed{A} <segment>1.5s-3.5s</segment>",
        "reasoning \\boxed{A} <segment>0s-2s</segment>",
        "reasoning \\boxed{A} <segment>2s-4s</segment>",
        "reasoning \\boxed{B} <segment>2s-5s</segment>",
        "reasoning \\boxed{B} <segment>2s-5s</segment>",
        "no boxed answer",
        "invalid \\boxed{}",
        "also invalid",
    ]
    scored = score_solver_responses(response_texts)
    assert scored["answer"] == "A", scored
    assert scored["majority_count"] == 5, scored
    assert scored["valid_answer_count"] == 7, scored
    assert scored["score"] == 0.5, scored
    assert _extract_candidate_segment("<segment>2.5s—5.0s</segment>") == [2.5, 5.0]
    assert _extract_candidate_segment("<segment>5s-2s</segment>") is None
    segment_filtered = score_solver_responses(
        [
            "\\boxed{<segment>1.0s-3.0s</segment>}",
            "\\boxed{B} <segment>1.0s-3.0s</segment>",
        ]
    )
    assert segment_filtered["candidate_answers"] == ["", "B"]
    assert segment_filtered["answer"] == "B"
    assert segment_filtered["score"] == 0.5

    with tempfile.TemporaryDirectory(prefix="evo-evaluate-contract-") as temp_dir:
        video_path = Path(temp_dir) / "full_video.pt"
        frames = torch.zeros((16, 3, 4, 4), dtype=torch.float32)
        torch.save(
            {
                "frames": frames,
                "metadata": {
                    "total_num_frames": 16,
                    "fps": 2.0,
                    "frames_indices": list(range(16)),
                    "width": 4,
                    "height": 4,
                    "duration": 8.0,
                },
            },
            video_path,
        )
        row = {
            "source_id": "video-1",
            "sample_id": "sample-1",
            "manifest_index": 0,
            "video_pt_path": str(video_path),
            "videos": ["video-1.mp4"],
            "window_start_frame": 2,
            "window_end_frame_exclusive": 10,
            "window_source_frame_indices": list(range(2, 10)),
            "segment_start_sec": 1.0,
            "segment_end_sec": 4.5,
            "question_type": "multiple choice",
            "question": "Which event happens after the object moves? A. X B. Y C. Z D. W",
            "answer": "B",
            "score": 0,
        }
        video_input = _build_video_input(
            row,
            processor=_FakeProcessor(),
            tokenizer=_FakeTokenizer(),
            prompt_template=Template(
                "<video>{{ content }} Return \\boxed{} and <segment>Xs-Ys</segment>."
            ),
        )
        assert video_input["prompt_token_ids"] == [101, 102, 103]
        loaded_frames = video_input["multi_modal_data"]["video"][0][0]
        assert int(loaded_frames.shape[0]) == 16

        evaluated = _common_result_fields(row)
        evaluated.update(scored)
        evaluated["num_candidates"] = len(response_texts)
        assert evaluated["questioner_answer"] == "B"
        assert evaluated["answer"] == "A"
        assert evaluated["target_segment"] == [1.0, 4.5]
        assert evaluated["preprocessed_video"] == str(video_path)

        curated, rejected = curate_rows([evaluated], min_score=0.3, max_score=0.8)
        assert rejected == []
        assert len(curated) == 1
        assert curated[0]["problem"] == row["question"]
        assert curated[0]["answer"] == "A"
        assert curated[0]["videos"] == ["video-1.mp4"]
        assert curated[0]["preprocessed_video"] == str(video_path)
        assert curated[0]["target_segment"] == [1.0, 4.5]

        invalid_majority = dict(evaluated)
        invalid_majority["problem_id"] = "segment-as-answer"
        invalid_majority["answer"] = "<segment>1.0s-3.0s</segment>"
        invalid_majority["score"] = 0.4
        guarded, rejected = curate_rows(
            [evaluated, invalid_majority], min_score=0.3, max_score=0.8
        )
        assert len(guarded) == 1
        assert len(rejected) == 1
        assert "<segment>" in rejected[0]["error"]

    print(
        {
            "candidate_denominator": len(response_texts),
            "majority_count": scored["majority_count"],
            "confidence": scored["score"],
            "full_video_frames": 16,
            "target_segment": [1.0, 4.5],
            "curated_rows": 1,
        }
    )


if __name__ == "__main__":
    main()

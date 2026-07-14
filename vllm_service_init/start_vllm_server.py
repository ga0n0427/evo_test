#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R-Zero Solver service with EasyVideoR1 preprocessed-video support.

The HTTP/file protocol is intentionally unchanged:

    GET /hello?name=/shared/tasks/tasks.json

The task file contains a list of records.  Legacy text tasks can still provide
``question`` and ``answer``.  EvoVid Questioner tasks instead provide:

    {
      "id": "...",
      "question": "...",
      "video_pt_path": "/shared/preprocessed/video.pt",
      "frame_order": "original" | "shuffle",
      "shuffle_seed": 123,
      "num_candidates": 10
    }

For video tasks this service reopens the original EasyVideoR1 artifact, creates
the shuffled view only in memory when requested, and writes a result with the
same ``id`` and ``frame_order``.  It never writes a second .pt artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import stopit
import torch
import vllm
from flask import Flask, jsonify, request
from mathruler.grader import extract_boxed_content, grade_answer
from transformers import AutoProcessor, AutoTokenizer
from transformers.video_utils import VideoMetadata


VIDEO_SOLVER_SYSTEM_PROMPT = (
    "You are a video reasoning solver. Answer the user's question using the video. "
    "Reason step by step, then put only the final answer within \\boxed{} so that "
    "independent answers can be compared."
)
_ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B-Base")
parser.add_argument(
    "--gpu_mem_util",
    type=float,
    default=0.8,
    help="The maximum GPU memory utilization fraction for vLLM.",
)
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--num_candidates", type=int, default=10)
args = parser.parse_args()


print("[init] Loading tokenizer, processor, and Solver model...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
processor = AutoProcessor.from_pretrained(args.model_path)
model = vllm.LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    gpu_memory_utilization=args.gpu_mem_util,
    disable_mm_preprocessor_cache=True,
    limit_mm_per_prompt={"video": 1},
)

sample_params = vllm.SamplingParams(
    max_tokens=args.max_tokens,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=args.num_candidates,
)


# Retain R-Zero's GPU idle behavior so the four standalone Solver services have
# the same lifecycle as before.
stop_event = threading.Event()
pause_event = threading.Event()


def gpu_idle_worker() -> None:
    print("[idle_worker] GPU idle worker started.")
    running = True
    while not stop_event.is_set():
        if pause_event.is_set():
            if running:
                print("[idle_worker] Paused.")
                running = False
            time.sleep(0.1)
            continue
        if not running:
            print("[idle_worker] Resumed.")
            running = True
        try:
            a = torch.rand((2000, 2000), dtype=torch.float32, device="cuda")
            b = torch.rand((2000, 2000), dtype=torch.float32, device="cuda")
            torch.matmul(a, b)
            torch.cuda.synchronize()
        except RuntimeError as exc:
            print(f"[idle_worker] Caught a RuntimeError: {exc}. Sleeping for 1s...")
            time.sleep(1)
    print("[idle_worker] GPU idle worker stopped.")


idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)
idle_thread.start()


@stopit.threading_timeoutable(default="TIMED_OUT")
def grade_answer_with_timeout(answer_a: str, answer_b: str) -> bool:
    """Keep the original R-Zero answer-equivalence fallback bounded."""
    return grade_answer(answer_a, answer_b)


def _metadata_to_vllm(metadata: dict[str, Any], frames: Any) -> VideoMetadata:
    frame_count = int(frames.shape[0]) if hasattr(frames, "shape") else len(frames)
    return VideoMetadata(
        total_num_frames=metadata.get("total_num_frames", frame_count),
        fps=metadata.get("fps"),
        frames_indices=metadata.get("frames_indices"),
        video_backend=metadata.get("video_backend"),
        width=metadata.get("width"),
        height=metadata.get("height"),
        duration=metadata.get("duration"),
    )


def _shuffle_frames(frames: Any, shuffle_seed: int) -> Any:
    """Permute only the frame axis; metadata stays attached to input positions."""
    frame_count = int(frames.shape[0]) if hasattr(frames, "shape") else len(frames)
    if frame_count < 2:
        return frames
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(shuffle_seed))
    permutation = torch.randperm(frame_count, generator=generator)
    if isinstance(frames, torch.Tensor):
        return frames.index_select(0, permutation)
    return [frames[index] for index in permutation.tolist()]


def _load_video_task(task: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    video_pt_path = task.get("video_pt_path")
    if not isinstance(video_pt_path, str) or not video_pt_path:
        raise ValueError("video task must contain a non-empty video_pt_path.")
    if not os.path.exists(video_pt_path):
        raise FileNotFoundError(f"Preprocessed video artifact not found: {video_pt_path}")

    artifact = torch.load(video_pt_path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or "frames" not in artifact or "metadata" not in artifact:
        raise KeyError(f"{video_pt_path} must contain 'frames' and 'metadata'.")
    frames = artifact["frames"]
    metadata = dict(artifact["metadata"])

    frame_order = task.get("frame_order", "original")
    if frame_order == "shuffle":
        shuffle_seed = task.get("shuffle_seed")
        if not isinstance(shuffle_seed, int):
            raise ValueError("shuffle task must contain an integer shuffle_seed.")
        frames = _shuffle_frames(frames, shuffle_seed)
    elif frame_order != "original":
        raise ValueError(f"Unsupported frame_order={frame_order!r}.")
    return frames, metadata


def _build_video_vllm_input(task: dict[str, Any]) -> dict[str, Any]:
    """Create the same preprocessed-video vLLM input shape used by EasyVideoR1."""
    question = task.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("video task must contain a non-empty question.")

    frames, metadata = _load_video_task(task)
    messages = [
        {"role": "system", "content": VIDEO_SOLVER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": question.strip()},
            ],
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    model_inputs = processor(
        text=[prompt],
        videos=[frames],
        add_special_tokens=False,
        video_metadata=[metadata],
        return_tensors="pt",
        do_resize=False,
        do_sample_frames=False,
    )
    return {
        "prompt_token_ids": model_inputs["input_ids"][0].tolist(),
        "multi_modal_data": {"video": [(frames, _metadata_to_vllm(metadata, frames))]},
        "mm_processor_kwargs": {"do_sample_frames": False, "do_resize": False},
    }


def _build_text_prompt(task: dict[str, Any]) -> str:
    """Legacy R-Zero text-task compatibility path."""
    question = task.get("question")
    answer = task.get("answer")
    if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
        raise ValueError("legacy text task requires non-empty question and answer.")
    messages = [
        {"role": "system", "content": VIDEO_SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _extract_candidate_answer(text: str) -> str:
    boxed = extract_boxed_content(text)
    if boxed:
        return str(boxed).strip()
    tags = list(_ANSWER_TAG_PATTERN.finditer(text))
    if tags:
        return tags[-1].group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _majority_answer(response: Any) -> tuple[str, float, list[str]]:
    candidates = [_extract_candidate_answer(output.text) for output in response.outputs]
    candidates = [candidate for candidate in candidates if candidate]
    answer_counts: dict[str, int] = {}
    for candidate in candidates:
        matched = False
        for existing in list(answer_counts):
            if candidate == existing or ("no " in candidate.lower() and "no " in existing.lower()):
                answer_counts[existing] += 1
                matched = True
                break
            try:
                equivalent = grade_answer_with_timeout(candidate, existing, timeout=10)
                if equivalent != "TIMED_OUT" and bool(equivalent):
                    answer_counts[existing] += 1
                    matched = True
                    break
                equivalent = grade_answer_with_timeout(existing, candidate, timeout=10)
                if equivalent != "TIMED_OUT" and bool(equivalent):
                    answer_counts[existing] += 1
                    matched = True
                    break
            except Exception as exc:
                print(f"[grader] comparison failed: {exc}")
        if not matched:
            answer_counts[candidate] = 1

    if not answer_counts:
        return "", 0.0, candidates
    answer = max(answer_counts, key=answer_counts.get)
    # Confidence is defined against the configured M candidates, as in R-Zero
    # and EvoVid.  Empty/unparseable candidates therefore lower confidence.
    confidence = answer_counts[answer] / len(response.outputs) if response.outputs else 0.0
    return answer, float(confidence), candidates


def _result_for_error(task: dict[str, Any], error: Exception | str) -> dict[str, Any]:
    return {
        "id": task.get("id"),
        "frame_order": task.get("frame_order", "original"),
        "question": task.get("question", ""),
        "answer": "",
        "score": -1,
        "results": [],
        "error": str(error),
    }


def _validate_candidate_counts(tasks: list[dict[str, Any]]) -> None:
    requested = {task.get("num_candidates", args.num_candidates) for task in tasks if "video_pt_path" in task}
    if not requested:
        return
    if requested != {args.num_candidates}:
        raise ValueError(
            "All video tasks must request the server's configured num_candidates "
            f"({args.num_candidates}); got {sorted(requested)!r}."
        )


app = Flask(__name__)


@app.route("/hello", methods=["GET"])
def hello():
    """Read a R-Zero task file, run text/video Solver generation, and save results."""
    pause_event.set()
    try:
        torch.cuda.synchronize()
        name = request.args.get("name")
        if not name:
            return jsonify({"error": "missing required query parameter: name"}), 400
        print(f"[server] Received request for task file: {name}")

        with open(name, "r", encoding="utf-8") as handle:
            tasks = json.load(handle)
        if not isinstance(tasks, list):
            return jsonify({"error": "task file must contain a JSON list"}), 400
        os.remove(name)
        _validate_candidate_counts(tasks)

        # Build video and legacy-text batches separately.  vLLM accepts either
        # input shape, but keeping them separate avoids mixed-modality behavior
        # differences across vLLM versions.
        video_indices: list[int] = []
        video_inputs: list[dict[str, Any]] = []
        text_indices: list[int] = []
        text_prompts: list[str] = []
        results_all: list[dict[str, Any] | None] = [None] * len(tasks)
        for index, task in enumerate(tasks):
            if not isinstance(task, dict):
                results_all[index] = _result_for_error({}, "task is not a JSON object")
                continue
            try:
                if "video_pt_path" in task:
                    video_input = _build_video_vllm_input(task)
                    video_indices.append(index)
                    video_inputs.append(video_input)
                else:
                    text_prompt = _build_text_prompt(task)
                    text_indices.append(index)
                    text_prompts.append(text_prompt)
            except Exception as exc:
                results_all[index] = _result_for_error(task, exc)

        responses_by_index: dict[int, Any] = {}
        if video_inputs:
            for index, response in zip(
                video_indices,
                model.generate(video_inputs, sampling_params=sample_params, use_tqdm=True),
            ):
                responses_by_index[index] = response
        if text_prompts:
            for index, response in zip(
                text_indices,
                model.generate(text_prompts, sampling_params=sample_params, use_tqdm=True),
            ):
                responses_by_index[index] = response

        for index, task in enumerate(tasks):
            if results_all[index] is not None:
                continue
            try:
                response = responses_by_index[index]
                answer, score, candidate_answers = _majority_answer(response)
                results_all[index] = {
                    "id": task.get("id"),
                    "frame_order": task.get("frame_order", "original"),
                    "question": task.get("question", ""),
                    "answer": answer,
                    "score": score,
                    "results": candidate_answers,
                }
            except Exception as exc:
                results_all[index] = _result_for_error(task, exc)

        out_path = Path(str(name).replace(".json", "_results.json"))
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(results_all, handle, ensure_ascii=False, indent=2)
        print(f"[server] Processed {name}, results saved to {out_path}.")
        return jsonify({"message": f"Processed {name}", "results_path": str(out_path)})
    except Exception as exc:
        print(f"[server] Request failed: {exc}")
        return jsonify({"error": str(exc)}), 500
    finally:
        pause_event.clear()


if __name__ == "__main__":
    try:
        app.run(host="127.0.0.1", port=args.port, threaded=True)
    finally:
        stop_event.set()
        idle_thread.join()
        print("[main] Application shutdown complete.")

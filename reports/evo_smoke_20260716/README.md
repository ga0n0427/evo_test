# EvoVid smoke-test outputs (2026-07-16)

이 디렉터리는 Qwen3-VL LoRA 연속학습, video-aware Solver 서버, 그리고 Solver 데이터용
8-frame window 생성 실험에서 나온 작은 결과 파일을 한곳에 모은 것이다.

## LoRA continuation

- `lora/verification_lora_only.json`: Stage A adapter를 Stage B에서 trainable로 다시 불러온 검증 결과
- `lora/stage_*_experiment_log.jsonl`: 두 단계의 학습 지표
- `lora/stage_*_generations.log`: 두 단계의 Questioner generation

검증 결과는 Stage B에서 이전 adapter 로드가 확인됐고, LoRA tensor 504/504개가 변경됐으며
positive gradient norm이 기록됐음을 보여준다.

## Video Solver server

- `video_server/reward_summary_before_prompt_fix.json`: expanded video token을 vLLM에 전달하던 실패 결과
- `video_server/reward_summary_after_prompt_fix.json`: raw video placeholder 수정 후 결과 (`solver_ok=10/10`)
- `video_server/reward_solver_results_20.json`: MMVU 10문제의 original/shuffle 20개 Solver 결과
- `video_server/questioner_text_task_*.json`: text-only HTTP 경로 확인 결과

## Balanced window generation

- `window_generation/synthetic_balanced8000_summary.json`: 초기 8-window 합성 분포 검사
- `window_generation/mmvu_raw_10.jsonl`: 실제 MMVU 10개 source record
- `window_generation/mmvu_preprocessed_16.jsonl`: 각 영상을 16 frames로 전처리한 record
- `window_generation/real18_manifest.jsonl`: 9개 window를 각 2개씩 배정한 실제 manifest
- `window_generation/real18_manifest_summary.json`: 18개 manifest 분포 요약
- `window_generation/real18_generation_results.json`: manifest를 거쳐 GPU로 생성한 18개 결과
- `window_generation/real9000_manifest_summary.json`: 목표 9,000개 manifest 분포 요약

최종 기본 분포는 Python end-exclusive slice 기준 `0:8`부터 `8:16`까지 9개 window이며,
9,000개 manifest에서 각 시작 위치가 정확히 1,000개다. 실제 MMVU 10개 영상은 각각 900회
배정됐고, GPU shard 8개는 각각 1,125개다.

18개 실제 GPU 입력은 모두 vLLM generation까지 성공했다. 사용한 smoke Questioner가
`<type>X</type><question>Y</question><answer>Z</answer>` 형식을 지키지 않아 parser 기준 유효
출력은 0/18이었으며, 원문 응답과 오류는 `real18_generation_results.json`에 보존했다.

## Excluded large/reproducible artifacts

다음 파일은 Git 크기를 불필요하게 늘리므로 포함하지 않았다.

- preprocessed `.pt` 영상 10개: 약 85 MiB
- merged/base model checkpoints
- 9,000행 manifest shard 8개: 약 3.2 MiB, summary와 입력 record로 재생성 가능
- 전체 vLLM/FSDP console logs

원본 실험 경로는 각 JSON에 기록되어 있으며 `/tmp` 정리 후에는 유효하지 않을 수 있다.

# EvoVid 간단 검증 결과 모음 (2026-07-16)

이 폴더에는 Qwen3-VL의 LoRA 이어학습, 영상 Solver 서버, 8프레임 문제 생성, Solver 평가와
upload 검증 결과가 들어 있다.

최근 Solver 평가 결과를 먼저 보려면 `question_evaluate/README.md`를 읽으면 된다. 각 점수의 의미와
문제별 결과를 쉬운 한글로 설명해 두었다.

## LoRA 이어학습

- `lora/verification_lora_only.json`: 앞 단계 LoRA를 다음 단계에서 다시 학습할 수 있는 상태로 불러왔는지 확인한 결과
- `lora/stage_*_experiment_log.jsonl`: 두 학습 단계의 수치 기록
- `lora/stage_*_generations.log`: 두 학습 단계의 Questioner 생성 결과

두 번째 단계가 첫 번째 단계의 LoRA를 실제로 불러왔고, LoRA tensor 504개가 모두 학습으로
변경된 것을 확인했다.

## 영상 Solver 서버

- `video_server/reward_summary_before_prompt_fix.json`: 영상 토큰을 잘못 전달했을 때의 실패 결과
- `video_server/reward_summary_after_prompt_fix.json`: 영상 연결 방식을 수정한 뒤의 성공 결과 (`solver_ok=10/10`)
- `video_server/reward_solver_results_20.json`: MMVU 10문제를 원본 영상과 섞은 영상으로 푼 20개 결과
- `video_server/questioner_text_task_*.json`: 영상 없는 텍스트 요청도 처리되는지 확인한 결과

## 8프레임 구간 균등 배정

- `window_generation/synthetic_balanced8000_summary.json`: 초기 8-window 합성 분포 검사
- `window_generation/mmvu_raw_10.jsonl`: 실제 MMVU 10개 source record
- `window_generation/mmvu_preprocessed_16.jsonl`: 각 영상을 16 frames로 전처리한 record
- `window_generation/real18_manifest.jsonl`: 9개 window를 각 2개씩 배정한 실제 manifest
- `window_generation/real18_manifest_summary.json`: 18개 manifest 분포 요약
- `window_generation/real18_generation_results.json`: manifest를 거쳐 GPU로 생성한 18개 결과
- `window_generation/real9000_manifest_summary.json`: 목표 9,000개 manifest 분포 요약

16프레임 안에서 `0:8`, `1:9`, ..., `8:16`까지 총 9가지 연속 구간을 사용한다. 9,000개를
만들면 각 시작 위치가 정확히 1,000번 사용된다. 실제 MMVU 영상 10개도 각각 900번씩 균등하게
배정됐고, GPU용 파일 8개에는 각각 1,125개가 들어갔다.

영상 18개는 모두 GPU 생성까지 성공했다. 다만 당시 사용한 Questioner가 정해진 출력 형식을
지키지 않아 정상 문제로 인정된 것은 0개였다. 잘못 나온 원문과 이유는
`real18_generation_results.json`에 그대로 보존했다.

## Solver 문제 평가와 upload

- `question_evaluate/README.md`: 사람이 읽기 쉬운 한글 설명
- `question_evaluate/mc4_input.json`: A/B/C/D 문제로 수정한 입력 4개
- `question_evaluate/mc4_results.json`: 전체 16프레임으로 문제당 10번씩 푼 상세 결과
- `question_evaluate/mc4_summary.json`: 주요 숫자를 한글로 정리한 요약
- `question_evaluate/mc4_train.json`: upload 직전에 만들어진 최종 학습 데이터

네 문제의 다수결 정답은 모두 기대 정답과 같았다. 점수는 `0.4, 0.4, 0.6, 0.5`로 네 문제 모두
학습 데이터 선택 범위에 들어왔다. 잘못된 `<segment>` 정답 후보는 제외했고, `upload.py`는 네 문제를
모두 학습 데이터로 만들면서 영상 경로와 정답 시간 구간을 보존했다. Hugging Face에는 올리지 않았다.

## Git에 넣지 않은 큰 파일

다음 파일은 Git 크기를 불필요하게 늘리므로 포함하지 않았다.

- preprocessed `.pt` 영상 10개: 약 85 MiB
- merged/base model checkpoints
- 9,000행 manifest shard 8개: 약 3.2 MiB, summary와 입력 record로 재생성 가능
- 전체 vLLM/FSDP console logs

원본 실험 경로는 각 JSON에 기록되어 있으며 `/tmp` 정리 후에는 유효하지 않을 수 있다.

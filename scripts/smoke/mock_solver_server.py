#!/usr/bin/env python3
"""CPU-only mock for the EvoVid file-backed Solver HTTP protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _unit_interval(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)


def _mock_score(task: dict) -> float:
    question = str(task.get("question", ""))
    original = 0.25 + 0.5 * _unit_interval(f"original:{question}")
    if task.get("frame_order", "original") == "shuffle":
        temporal_gap = 0.05 + 0.25 * _unit_interval(f"shuffle:{question}")
        return max(0.0, original - temporal_gap)
    return original


class MockSolverHandler(BaseHTTPRequestHandler):
    server_version = "EvoVidMockSolver/1.0"

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if parsed.path != "/hello":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        names = parse_qs(parsed.query).get("name", [])
        if len(names) != 1 or not names[0]:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "missing name"})
            return

        task_path = Path(names[0])
        result_path = Path(str(task_path).replace(".json", "_results.json"))
        try:
            with task_path.open(encoding="utf-8") as handle:
                tasks = json.load(handle)
            if not isinstance(tasks, list):
                raise TypeError("task file must contain a JSON list")

            results = []
            for task in tasks:
                if not isinstance(task, dict):
                    raise TypeError("every task must be a JSON object")
                score = _mock_score(task)
                results.append(
                    {
                        "id": task.get("id"),
                        "frame_order": task.get("frame_order", "original"),
                        "question": task.get("question", ""),
                        "answer": "MOCK",
                        "score": score,
                        "results": ["MOCK"],
                    }
                )

            task_path.unlink(missing_ok=True)
            with result_path.open("w", encoding="utf-8") as handle:
                json.dump(results, handle, ensure_ascii=False, indent=2)
            print(f"Processed {len(tasks)} tasks -> {result_path}", flush=True)
            self._send_json(
                HTTPStatus.OK,
                {"message": "processed by mock solver", "results_path": str(result_path)},
            )
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MockSolverHandler)
    print(f"Mock Solver listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

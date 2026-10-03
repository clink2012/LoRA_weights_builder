"""Temporary loopback-only design questionnaire; standard library only."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
import re
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from uuid import uuid4


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_DATA = PROJECT / ".local" / "gui-questionnaire"
STATIC_FILES = {"/": "index.html", "/index.html": "index.html",
                "/styles.css": "styles.css", "/app.js": "app.js", "/data.js": "data.js"}
MAX_BODY = 65536


def validate_payload(value: object) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Unrecognised questionnaire format.")
    answers = value.get("answers")
    notes = value.get("notes", "")
    submitted = value.get("submitted", False)
    if not isinstance(answers, dict) or len(answers) > 40:
        raise ValueError("Answers must be a small set of questionnaire choices.")
    for key, answer in answers.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", key):
            raise ValueError("Unrecognised question identifier.")
        items = answer if isinstance(answer, list) else [answer]
        if len(items) > 32 or not all(isinstance(item, str) and len(item) <= 500 for item in items):
            raise ValueError("A choice is too long or has an invalid format.")
    if not isinstance(notes, str) or len(notes) > 10000 or not isinstance(submitted, bool):
        raise ValueError("Notes or completion state have an invalid format.")
    return {"schema_version": 1, "answers": answers, "notes": notes, "submitted": submitted}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".answer-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class QuestionnaireServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, data_root: Path = DEFAULT_DATA):
        self.data_root = data_root.resolve()
        self.save_lock = threading.Lock()
        super().__init__(("127.0.0.1", port), Handler)


class Handler(BaseHTTPRequestHandler):
    server: QuestionnaireServer

    def log_message(self, fmt: str, *args: object) -> None:
        # No response content or notes in runtime logs.
        super().log_message(fmt, *args)

    def send_bytes(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def json_response(self, code: int, value: dict) -> None:
        self.send_bytes(code, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def local_request(self) -> bool:
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            self.json_response(403, {"error": "Use the local questionnaire address."})
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in hosts}:
            self.json_response(403, {"error": "This request did not come from the local questionnaire."})
            return False
        return True

    def do_GET(self) -> None:
        if not self.local_request():
            return
        route = urlsplit(self.path).path
        if route == "/api/health":
            self.json_response(200, {"app": "lora-gui-questionnaire", "schema_version": 1})
            return
        if route == "/api/answers":
            path = self.server.data_root / "answers.json"
            with self.server.save_lock:
                try:
                    if path.exists():
                        stored = json.loads(path.read_text(encoding="utf-8"))
                        value = validate_payload(stored)
                        value["saved_at"] = stored.get("saved_at")
                        if value["saved_at"] is not None and not isinstance(value["saved_at"], str):
                            raise ValueError("Invalid saved timestamp.")
                    else:
                        value = {"schema_version": 1, "answers": {}, "notes": "", "submitted": False, "saved_at": None}
                except (OSError, ValueError):
                    self.json_response(500, {"error": "Saved answers could not be read. They have not been replaced."})
                    return
            self.json_response(200, value)
            return
        if route in STATIC_FILES:
            path = HERE / STATIC_FILES[route]
        elif re.fullmatch(r"/references/[1-7]\.png", route):
            path = self.server.data_root / "reference-images" / route.rsplit("/", 1)[1]
        else:
            self.json_response(404, {"error": "Page not found."})
            return
        try:
            body = path.read_bytes()
        except OSError:
            self.json_response(404, {"error": "File not available."})
            return
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix in {".js", ".html", ".css"}:
            mime = {".js": "text/javascript", ".html": "text/html", ".css": "text/css"}[path.suffix] + "; charset=utf-8"
        self.send_bytes(200, body, mime)

    def do_POST(self) -> None:
        # Read the bounded body before returning a rejection. Closing a Windows
        # socket with unread request bytes can reset it before the browser sees
        # the error response.
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                raise ValueError("The answer submission is empty or too large.")
            body = self.rfile.read(length)
        except ValueError as error:
            self.json_response(400, {"error": str(error)})
            return
        if not self.local_request():
            return
        if urlsplit(self.path).path != "/api/answers":
            self.json_response(404, {"error": "Page not found."})
            return
        if self.headers.get_content_type() != "application/json":
            self.json_response(415, {"error": "Answers must be sent as JSON."})
            return
        try:
            value = validate_payload(json.loads(body))
        except (ValueError, UnicodeError) as error:
            self.json_response(400, {"error": str(error)})
            return
        now = datetime.now(timezone.utc)
        value["saved_at"] = now.isoformat()
        receipt = None
        with self.server.save_lock:
            try:
                if value["submitted"]:
                    receipt = f"completed-{now:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}.json"
                    atomic_json(self.server.data_root / "submissions" / receipt, value)
                atomic_json(self.server.data_root / "answers.json", value)
            except OSError:
                self.json_response(500, {"error": "Could not save answers. Keep this page open and retry."})
                return
        self.json_response(200, {"ok": True, "saved_at": value["saved_at"], "submitted": value["submitted"], "receipt": receipt})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5186)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    server = QuestionnaireServer(args.port, args.data_dir)
    print(f"LoRA design questionnaire: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

"""evals/learning_loop/gpu_study.py: downloads are kept only when their sha256 matches, the engine is the pinned
one, the agent's account is refused when it can read a protected path, checkpoints copy what changed and are restored
only onto a machine without the run, and the pinned preset agrees with the runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

import pytest

from evals.learning_loop import gpu_study


def test_a_download_is_kept_only_when_its_sha256_matches(tmp_path):
    source = tmp_path / "artifact.tar.gz"
    source.write_bytes(b"pinned bytes")
    good = hashlib.sha256(b"pinned bytes").hexdigest()
    with pytest.raises(SystemExit, match="expected"):
        gpu_study.download(source.as_uri(), tmp_path / "bad", "0" * 64)
    assert not (tmp_path / "bad").exists() and not (tmp_path / "bad.part").exists()
    gpu_study.download(source.as_uri(), tmp_path / "good", good, size=len(b"pinned bytes"))
    assert (tmp_path / "good").read_bytes() == b"pinned bytes"


def test_the_engine_refuses_a_checkout_that_pins_another_llama_cpp(tmp_path, monkeypatch):
    (tmp_path / "pm").mkdir()
    lock = {"packages": {"llamacpp-cuda": {"version": "11672", "artifacts": {"linux-x64": []}}}}
    (tmp_path / "pm" / "lock.json").write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(gpu_study, "REPO", tmp_path)
    monkeypatch.setattr(gpu_study, "STUDY_ROOT", tmp_path / "study")
    with pytest.raises(SystemExit, match="b11672, not the pre-registered b11671"):
        gpu_study.cmd_engine(argparse.Namespace())
    assert not (tmp_path / "study").exists()  # refused before anything was downloaded


def test_the_isolation_check_fails_closed():
    def answers(*codes):
        replies = iter(codes)
        return lambda cmd, timeout: subprocess.CompletedProcess(cmd, next(replies))
    assert gpu_study.readable_by("agent", "/content/drive", run=answers(1, 0))  # ls works: exposed
    assert gpu_study.readable_by("agent", "/content/drive", run=answers(0))  # test -r works: exposed
    assert not gpu_study.readable_by("agent", "/content/drive", run=answers(1, 2))


def test_a_checkpoint_copies_new_and_changed_files_only(tmp_path):
    src, dst = tmp_path / "run", tmp_path / "checkpoint"
    (src / "cell").mkdir(parents=True)
    (src / "results.jsonl").write_text("row 1\n", encoding="utf-8")
    (src / "cell" / "home.txt").write_text("state", encoding="utf-8")
    assert gpu_study.mirror(src, dst) == 2
    assert gpu_study.mirror(src, dst) == 0
    (src / "results.jsonl").write_text("row 1\nrow 2\n", encoding="utf-8")
    os.utime(src / "results.jsonl", (1, 1))
    assert gpu_study.mirror(src, dst) == 1
    assert (dst / "results.jsonl").read_text(encoding="utf-8") == "row 1\nrow 2\n"


def test_a_checkpoint_is_restored_only_onto_a_machine_without_a_run(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "run").mkdir(parents=True)
    (checkpoint / "run" / "results.jsonl").write_text("row 1\n", encoding="utf-8")
    monkeypatch.setattr(gpu_study, "run_dir", lambda study: tmp_path / "runs" / study)
    monkeypatch.setattr(gpu_study, "_run", lambda cmd, timeout: subprocess.CompletedProcess(cmd, 0))  # chown
    args = argparse.Namespace(checkpoint=str(checkpoint), study="calibration")
    assert gpu_study.cmd_restore(args)["restored_files"] == 1  # a new machine
    local = tmp_path / "runs" / "calibration" / "results.jsonl"
    local.write_text("row 1\nrow 2\n", encoding="utf-8")  # the run went on before its next checkpoint copy
    assert gpu_study.cmd_restore(args)["restored_files"] == 0
    assert local.read_text(encoding="utf-8") == "row 1\nrow 2\n"


def test_the_pinned_preset_is_the_one_the_runner_serves():
    text = gpu_study.PINNED_PRESET.read_text(encoding="utf-8")
    assert f"[{gpu_study.MODEL_ID}]" in text
    assert f"model = {gpu_study.model_path()}" in text
    assert gpu_study.context_length() == 98304
    assert "override-tensor" not in text  # the whole model on the card, nothing pushed to the CPU


def _server_reporting(n_ctx):
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body):
            raw = _json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self._send({"choices": [{"message": {"content": "hi"}}]})

        def do_GET(self):
            self._send({"default_generation_settings": {"n_ctx": n_ctx}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_serving_stops_when_a_session_would_get_less_than_the_preset_window():
    split = _server_reporting(24576)  # the window shared between the router's four slots
    try:
        with pytest.raises(SystemExit, match="24576"):
            gpu_study._load_model(split.server_address[1], "key")
    finally:
        split.shutdown()
    whole = _server_reporting(98304)
    try:
        assert gpu_study._load_model(whole.server_address[1], "key") == 98304
    finally:
        whole.shutdown()

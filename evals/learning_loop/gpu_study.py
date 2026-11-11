#!/usr/bin/env python3
"""Run the pre-registered learning-loop study on one GPU machine (Colab or Kaggle).

Every step pins what it uses and stops when something differs:

  engine   Hermes' pinned llama.cpp (pm/lock.json, CUDA linux-x64) checked by sha256, or the same commit built from
           source when this machine's driver cannot run that binary
  model    the pinned GGUF from Hugging Face, checked by size and sha256
  preset   Hermes' own launch preset for a 24 GiB card (hermes_cli/local_runtime/presets.py), generated from the
           downloaded file and compared byte for byte with the pre-registered study_preset.ini
  serve    llama-server started the way Hermes' supervisor starts it, on 127.0.0.1, model loaded before it returns
  user     the unprivileged account the agent runs as, refused if it can read a protected path such as Drive
  agent-env  Hermes' environment for that account, and the interpreter of the environment it committed
  restore  the checkpoint copied back before a resumed block
  run      one block of repetitions as that account (live.py), mirrored to the checkpoint every few minutes

Only "preset" imports Hermes, so it runs inside the Hermes environment. Everything else is the standard library and
runs as root. See PREREGISTRATION.md beside this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# One fixed place on every machine, so the preset's model path, and with it the preset itself, is identical.
STUDY_ROOT = Path("/opt/hermes-study")
MODEL = {"repo": "unsloth/Qwen3.8-27B-GGUF", "revision": "4ca720788d1e01f1bff70c033e0d0028fd02e502",
         "file": "Qwen3.8-27B-UD-Q4_K_M.gguf", "size": 16_464_440_224,
         "sha256": "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482"}
MODEL_ID = "Qwen3.8-27B-UD-Q4_K_M"  # the preset section name, which requests must use
ENGINE_TAG, ENGINE_COMMIT = "b11671", "ba3c05f71da85cd0439e84ba42b0478b24d5db16"
ENGINE_SOURCE = "https://github.com/ggml-org/llama.cpp"
# The card the plan is made for: 24 GiB as nvidia-smi reports an RTX 4090, beside 32 GiB of RAM. Every GPU in the
# study runs this same plan, so outputs do not depend on which machine served them.
REFERENCE_VRAM_MIB, REFERENCE_RAM_GIB = 24564, 32
PINNED_PRESET = Path(__file__).with_name("study_preset.ini")
AGENT_USER = "hermes-eval"
PORT = 18081
CHUNK = 8 << 20
# Socket-level stalls only: a healthy download or build never waits this long for one read.
DOWNLOAD_TIMEOUT_S = 120
GIT_TIMEOUT_S = 600
BUILD_TIMEOUT_S = 5400
PROBE_TIMEOUT_S = 120
HEALTH_WAIT_S = 300
LOAD_TIMEOUT_S = 1200
SYNC_EVERY_S = 300
EPISODE_TIMEOUT_S = 3600
# Episode 1 is the session Hermes learns from, so it is the task that took the most tool iterations in the first
# calibration, and it runs until the background review writes something (PREREGISTRATION.md, Deviation 1).
EPISODE_ORDER = ("currency_symbol", "percent_of", "skip_empty_lines", "discounted_total", "parse_symbols")
LEARN_FIRST_TRIES = 5


def _run(cmd: list, timeout: int, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout, check=False,
                          **kwargs)


def _state(name: str) -> dict:
    path = STUDY_ROOT / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_state(name: str, data: dict) -> None:
    STUDY_ROOT.mkdir(parents=True, exist_ok=True)
    (STUDY_ROOT / f"{name}.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, sha256: str, size: int | None = None) -> None:
    """Stream *url* to *dest*, hashing as it goes. The file is kept only if its size and sha256 match."""
    if dest.exists() and (size is None or dest.stat().st_size == size) and sha256_of(dest) == sha256:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part, digest, received = dest.with_name(dest.name + ".part"), hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response, part.open("wb") as out:
        while chunk := response.read(CHUNK):
            digest.update(chunk)
            out.write(chunk)
            received += len(chunk)
    if (size is not None and received != size) or digest.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise SystemExit(f"{dest.name}: got {received} bytes with sha256 {digest.hexdigest()}, "
                         f"expected {size or '?'} bytes with sha256 {sha256}")
    part.replace(dest)


# ── engine ────────────────────────────────────────────────────────────────────

def _gpu_facts() -> list[dict]:
    fields = ("name", "memory.total", "driver_version", "compute_cap")
    proc = _run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"], PROBE_TIMEOUT_S)
    if proc.returncode != 0:
        raise SystemExit(f"no NVIDIA GPU here: {proc.stderr.strip()[:300]}")
    return [dict(zip(fields, (v.strip() for v in line.split(",")))) for line in proc.stdout.strip().splitlines()]


def _library_path(engine_dir: Path) -> str:
    dirs = sorted({str(p.parent) for p in engine_dir.rglob("*.so*")})
    return os.pathsep.join([*dirs, os.environ.get("LD_LIBRARY_PATH", "")])


def _cuda_devices(binary: Path, engine_dir: Path) -> list[str]:
    """The CUDA devices this build can drive; empty when the driver is too old for it."""
    proc = _run([binary, "--list-devices"], PROBE_TIMEOUT_S, env={**os.environ, "LD_LIBRARY_PATH": _library_path(engine_dir)})
    return [line.strip() for line in (proc.stdout + proc.stderr).splitlines() if line.strip().startswith("CUDA")]


def _cuda_version(text: str) -> tuple[int, int]:
    """The first "CUDA Version: X.Y" (nvidia-smi) or "cuda-X.Y" (a release name) in *text*, (0, 0) if none."""
    match = re.search(r"(?:CUDA Version:\s*|cuda-)(\d+)\.(\d+)", text)
    return (int(match[1]), int(match[2])) if match else (0, 0)


def _build_from_source(target: Path) -> tuple[Path, str]:
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        raise SystemExit("no CUDA toolkit (nvcc) here to build llama.cpp with")
    toolkit = re.search(r"release (\d+\.\d+)", _run([nvcc, "--version"], PROBE_TIMEOUT_S).stdout)
    src = target / "llama.cpp"
    if not src.exists():
        _run(["git", "clone", "--depth", "1", "--branch", ENGINE_TAG, ENGINE_SOURCE, src], GIT_TIMEOUT_S)
    head = _run(["git", "-C", src, "rev-parse", "HEAD"], GIT_TIMEOUT_S).stdout.strip()
    if head != ENGINE_COMMIT:
        raise SystemExit(f"llama.cpp {ENGINE_TAG} resolved to {head}, not the pinned {ENGINE_COMMIT}")
    archs = ";".join(sorted({g["compute_cap"].replace(".", "") for g in _gpu_facts()}))
    for cmd in (["cmake", "-S", src, "-B", src / "build", "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={archs}",
                 f"-DCMAKE_CUDA_COMPILER={nvcc}", "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF"],
                ["cmake", "--build", src / "build", "--target", "llama-server", "-j", str(os.cpu_count() or 4)]):
        proc = _run(cmd, BUILD_TIMEOUT_S)
        if proc.returncode != 0:
            raise SystemExit(f"building llama.cpp failed:\n{(proc.stdout + proc.stderr)[-3000:]}")
    how = f"built from {ENGINE_SOURCE} at {ENGINE_COMMIT} with CUDA {toolkit[1] if toolkit else '?'} for sm_{archs}"
    return src / "build" / "bin" / "llama-server", how


def _unpack(artifact: Path, into: Path) -> None:
    """A release tarball, or the libgomp .deb Hermes installs beside the engine. dpkg-deb -x only extracts the
    files and runs no package script, and it reads the zstd payload this machine's Python may not."""
    into.mkdir(parents=True, exist_ok=True)
    if artifact.name.endswith(".deb"):
        proc = _run(["dpkg-deb", "-x", artifact, into / artifact.stem], PROBE_TIMEOUT_S)
        if proc.returncode != 0:
            raise SystemExit(f"could not unpack {artifact.name}: {proc.stderr.strip()[:300]}")
        return
    with tarfile.open(artifact) as archive:
        archive.extractall(into, filter="data")


def cmd_engine(_args: argparse.Namespace) -> dict:
    """Hermes' pinned prebuilt engine when this driver runs it, else the same commit built from source."""
    package = json.loads((REPO / "pm" / "lock.json").read_text(encoding="utf-8"))["packages"]["llamacpp-cuda"]
    if f"b{package['version']}" != ENGINE_TAG:  # a later checkout could pin another engine
        raise SystemExit(f"pm/lock.json pins llama.cpp b{package['version']}, not the pre-registered {ENGINE_TAG}")
    prebuilt, artifacts = STUDY_ROOT / "engine" / "prebuilt", package["artifacts"]["linux-x64"]
    driver, built_with = _cuda_version(_run(["nvidia-smi"], PROBE_TIMEOUT_S).stdout), _cuda_version(artifacts[0]["url"])
    # A driver older than the build's toolkit runs it only where the build ships the card's own machine code, since
    # it cannot compile the build's PTX. On Colab's A100 (driver 580.82, CUDA 13.0) this CUDA 13.4 build listed the
    # card, then aborted loading the model: "the provided PTX was compiled with an unsupported toolchain".
    binary = None
    if driver >= built_with:
        for artifact in artifacts:
            local = STUDY_ROOT / "downloads" / artifact["url"].rsplit("/", 1)[1]
            download(artifact["url"], local, artifact["sha256"])
            _unpack(local, prebuilt)
        binary = next(prebuilt.rglob("llama-server"), None)
    how, engine_dir = "prebuilt (pm/lock.json)", prebuilt
    if binary is None or not _cuda_devices(binary, prebuilt):
        engine_dir, artifacts = STUDY_ROOT / "engine" / "source", []
        binary, how = _build_from_source(engine_dir)
    devices = _cuda_devices(binary, engine_dir)
    if not devices:
        raise SystemExit("the engine sees no CUDA device")
    state = {"tag": ENGINE_TAG, "commit": ENGINE_COMMIT, "how": how, "artifacts": artifacts,
             "driver_cuda": ".".join(map(str, driver)), "prebuilt_cuda": ".".join(map(str, built_with)),
             "binary": str(binary), "engine_dir": str(engine_dir), "sha256": sha256_of(binary),
             "devices": devices, "gpus": _gpu_facts()}
    _save_state("engine", state)
    return state


# ── model and preset ─────────────────────────────────────────────────────────

def model_path() -> Path:
    return STUDY_ROOT / "models" / MODEL["file"]


def cmd_model(_args: argparse.Namespace) -> dict:
    url = f"https://huggingface.co/{MODEL['repo']}/resolve/{MODEL['revision']}/{MODEL['file']}"
    download(url, model_path(), MODEL["sha256"], MODEL["size"])
    model_path().chmod(0o644)
    for directory in (STUDY_ROOT, model_path().parent):
        directory.chmod(0o755)
    return {k: MODEL[k] for k in ("repo", "revision", "file", "sha256")}


def reference_preset(gguf: Path) -> str:
    """The preset Hermes' local runtime writes for this file on the reference card."""
    from hermes_cli.local_runtime import hardware
    from hermes_cli.local_runtime.estimator import HardwareBudget
    from hermes_cli.local_runtime.presets import preset_for_model, render_presets
    total = REFERENCE_VRAM_MIB << 20
    margin = max(hardware._MARGIN_FLOOR, int(total * hardware._MARGIN_FRACTION))
    budget = HardwareBudget(usable_vram_bytes=total - margin, total_device_bytes=total,
                            ram_available_bytes=REFERENCE_RAM_GIB << 30, gpu_name="24 GiB reference card",
                            platform="linux")
    entry = preset_for_model(gguf, budget, set(), engine_tag=ENGINE_TAG)
    if entry is None or entry.refusal or entry.spilled:
        raise SystemExit(f"Hermes would not serve {gguf.name} whole on the reference card: {entry}")
    return render_presets([entry])


def cmd_preset(args: argparse.Namespace) -> dict:
    text = reference_preset(model_path())
    Path(args.out).write_text(text, encoding="utf-8")
    pinned = PINNED_PRESET.read_text(encoding="utf-8")
    if text != pinned:
        raise SystemExit(f"the generated preset differs from the pre-registered {PINNED_PRESET.name}:\n{text}")
    return {"preset": args.out, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def context_length() -> int:
    """The window in the pre-registered preset (its ctx-size)."""
    for line in PINNED_PRESET.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "ctx-size":
            return int(value)
    raise SystemExit(f"{PINNED_PRESET.name} has no ctx-size")


# ── server ────────────────────────────────────────────────────────────────────

def _direct_io_args(binary: Path, engine_dir: Path) -> list[str]:
    """The loading option this engine supports, as hermes_cli/local_runtime/supervisor.py picks it."""
    proc = _run([binary, "--help"], PROBE_TIMEOUT_S, env={**os.environ, "LD_LIBRARY_PATH": _library_path(engine_dir)})
    text = proc.stdout + proc.stderr
    if "--load-mode" in text:
        return ["--load-mode", "dio"]
    return ["-dio"] if "--direct-io" in text else []


def _answers(port: int) -> bool:
    """Whether anything answers HTTP on *port*. Any status counts: llama-server says 503 while it loads a model."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5):
            return True
    except urllib.error.HTTPError:
        return True
    except OSError:
        return False


def _wait_healthy(port: int, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + HEALTH_WAIT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"llama-server exited with {process.returncode}, see {STUDY_ROOT / 'server.log'}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    raise SystemExit("llama-server did not become healthy")


def _request(port: int, key: str, path: str, body: dict | None = None, timeout: int = PROBE_TIMEOUT_S) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        raise SystemExit(f"llama-server answered {path} with HTTP {error.code}: "
                         f"{error.read().decode('utf-8', 'replace')[:500]}") from error


def _load_model(port: int, key: str) -> int:
    """Load the model with one token, then read the window one session gets, the way Hermes reads it
    (/props?model=, agent/model_metadata.py). The router keeps several slots, and a window split between them
    would leave a session too little room, so anything but the preset's full window stops here."""
    _request(port, key, "/v1/chat/completions", {"model": MODEL_ID, "max_tokens": 1,
                                                 "messages": [{"role": "user", "content": "hi"}]}, LOAD_TIMEOUT_S)
    props = _request(port, key, f"/props?model={MODEL_ID}")
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
    if n_ctx != context_length():
        raise SystemExit(f"a session gets {n_ctx} tokens, not the preset's {context_length()}")
    return n_ctx


def cmd_serve(args: argparse.Namespace) -> dict:
    engine = _state("engine") or cmd_engine(args)
    binary, engine_dir = Path(engine["binary"]), Path(engine["engine_dir"])
    preset, key_file = STUDY_ROOT / "presets.ini", STUDY_ROOT / "server.key"
    cmd = [str(binary), "--host", "127.0.0.1", "--port", str(PORT), "--api-key-file", str(key_file),
           "--models-max", "1", "--models-autoload", "--metrics", "--slots", "--no-ui", "--jinja",
           *_direct_io_args(binary, engine_dir), "--models-preset", str(preset)]
    if _answers(PORT):  # Run all again on this machine: keep the server this step started, if it is this one
        running = _state("server")
        if running.get("command") != cmd or sha256_of(preset) != sha256_of(PINNED_PRESET):
            raise SystemExit(f"another server answers on port {PORT}: Runtime, Disconnect and delete runtime, "
                             "then Run all")
        return {**running, "session_n_ctx": _load_model(PORT, key_file.read_text(encoding="utf-8").strip())}
    shutil.copyfile(PINNED_PRESET, preset)
    if not key_file.exists():
        key_file.write_text(hashlib.sha256(os.urandom(32)).hexdigest(), encoding="utf-8")
    shutil.chown(key_file, group=AGENT_USER)
    key_file.chmod(0o640)
    log = (STUDY_ROOT / "server.log").open("a", encoding="utf-8")
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                               env={**os.environ, "LD_LIBRARY_PATH": _library_path(engine_dir)})
    _wait_healthy(PORT, process)
    n_ctx = _load_model(PORT, key_file.read_text(encoding="utf-8").strip())
    state = {"pid": process.pid, "port": PORT, "command": cmd, "model_id": MODEL_ID,
             "preset_sha256": sha256_of(preset), "context_length": context_length(), "session_n_ctx": n_ctx}
    _save_state("server", state)
    return state


# ── the agent's account ──────────────────────────────────────────────────────

def readable_by(user: str, path: str, run=_run) -> bool:
    """True when *user* can list or read *path*."""
    return any(run(["runuser", "-u", user, "--", *probe, path], PROBE_TIMEOUT_S).returncode == 0
               for probe in (["test", "-r"], ["ls"]))


def cmd_user(args: argparse.Namespace) -> dict:
    if _run(["id", "-u", AGENT_USER], PROBE_TIMEOUT_S).returncode != 0:
        proc = _run(["useradd", "--create-home", "--shell", "/bin/bash", AGENT_USER], PROBE_TIMEOUT_S)
        if proc.returncode != 0:
            raise SystemExit(f"could not create {AGENT_USER}: {proc.stderr.strip()}")
    # Hermes finds the environment it committed for a checkout under the folder the checkout sits in, when the
    # checkout is <data root>/hermes-agent as in a standard install. Each episode moves HOME into its own sandbox,
    # so that is the only route that works. The account may create files there (its installs/), but the sticky
    # bit stops it renaming or deleting root's read-only checkout.
    if REPO.name != "hermes-agent":
        raise SystemExit(f"clone the repository as <data root>/hermes-agent, not {REPO}")
    shutil.chown(REPO.parent, user="root", group=AGENT_USER)
    REPO.parent.chmod(0o1775)
    # Hermes' environment setup also writes its launchers into <repo>/.hermes/bin. That one folder belongs to the
    # account. The rest of the checkout, harness and grader included, stays read-only to it.
    launchers = REPO / ".hermes"
    launchers.mkdir(exist_ok=True)
    _run(["chown", "-R", f"{AGENT_USER}:{AGENT_USER}", launchers], PROBE_TIMEOUT_S)
    exposed = [p for p in args.protect if os.path.exists(p) and readable_by(AGENT_USER, p)]
    if exposed:
        raise SystemExit(f"{AGENT_USER} can read {exposed}: the agent would reach them, so nothing runs")
    return {"user": AGENT_USER, "protected": args.protect}


# ── run, checkpoint, restore ─────────────────────────────────────────────────

def run_dir(study: str) -> Path:
    return Path(f"/home/{AGENT_USER}/runs/{study}")


def mirror(src: Path, dst: Path) -> int:
    """Copy every file that is new or changed (size or mtime) from *src* into *dst*; return how many."""
    copied = 0
    for path in src.rglob("*"):
        target = dst / path.relative_to(src)
        if path.is_symlink() or not path.is_file():
            continue
        stat = path.stat()
        if target.exists() and target.stat().st_size == stat.st_size and target.stat().st_mtime == stat.st_mtime:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def cmd_restore(args: argparse.Namespace) -> dict:
    """Copy the checkpoint back onto a machine that has no run yet. A run this machine already has is never older
    than its checkpoint, which is only ever copied from it, so it is left as it is."""
    source, target = Path(args.checkpoint) / "run", run_dir(args.study)
    if target.exists():
        return {"restored_files": 0, "run": str(target), "kept": "the run already on this machine"}
    copied = mirror(source, target) if source.exists() else 0
    _run(["chown", "-R", f"{AGENT_USER}:{AGENT_USER}", target.parent], PROBE_TIMEOUT_S)
    return {"restored_files": copied, "run": str(target)}


# A machine behind a proxy (some CI sandboxes) needs its proxy and CA settings in the agent's account too. Colab and
# Kaggle set none of these, so there nothing is passed.
PROXY_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy", "UV_NATIVE_TLS")
CA_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")


def network_env() -> list[str]:
    env = [f"{name}={os.environ[name]}" for name in PROXY_VARS if os.environ.get(name)]
    bundle = next((os.environ[name] for name in CA_VARS if os.environ.get(name)), None)
    if bundle:
        shared = STUDY_ROOT / "ca-bundle.crt"  # the original may sit under /root, which the account cannot enter
        shutil.copyfile(bundle, shared)
        shared.chmod(0o644)
        env += [f"{name}={shared}" for name in CA_VARS]
    return env


PROJECT_PYTHON = ("from pathlib import Path; from pm.environments import project_python; "
                  "print(project_python(Path('.').resolve()))")


def as_agent(command: list[str], *, through_wrapper: bool = False) -> list[str]:
    """*command* run from the checkout as the agent's account, with nothing else inherited."""
    home = f"/home/{AGENT_USER}"
    env = [f"HOME={home}", f"HERMES_HOME={REPO.parent}", f"HERMES_RUNTIME_DIR={home}/.hermes-runtime",
           "PATH=/usr/local/bin:/usr/bin:/bin", "LANG=C.UTF-8", *network_env()]
    launch = f"scripts/run-in-hermes-env {shlex.join(command)}" if through_wrapper else shlex.join(command)
    return ["runuser", "-u", AGENT_USER, "--", "env", "-i", *env, "bash", "-c",
            f"cd {shlex.quote(str(REPO))} && exec {launch}"]


def cmd_agent_env(_args: argparse.Namespace) -> dict:
    """Set up Hermes' environment for the agent's account and record its interpreter.

    That is the interpreter of the environment Hermes committed for this checkout, the one Hermes' own evals run
    their workers on (evals/prefix_rediet). Unlike the bare store Python behind scripts/run-in-hermes-env, it
    boots Hermes from any HOME, which every episode's sandbox needs."""
    process = subprocess.run(as_agent(["python", "-c", PROJECT_PYTHON], through_wrapper=True), capture_output=True,
                             text=True, timeout=BUILD_TIMEOUT_S, check=False)
    lines = process.stdout.strip().splitlines()
    if process.returncode != 0 or not lines or not Path(lines[-1]).is_file():
        raise SystemExit(f"Hermes' environment did not set up for {AGENT_USER}:\n"
                         f"{(process.stdout + process.stderr)[-3000:]}")
    state = {"python": lines[-1]}
    _save_state("agent", state)
    return state


def agent_python() -> str:
    return (_state("agent") or cmd_agent_env(argparse.Namespace()))["python"]


def live_command(args: argparse.Namespace, server: dict) -> list[str]:
    model = f"{MODEL['file']} sha256:{MODEL['sha256']} on llama.cpp {ENGINE_TAG} ({ENGINE_COMMIT[:9]})"
    return as_agent([agent_python(), "evals/learning_loop/live.py",
            "--key-file", str(STUDY_ROOT / "server.key"), "--out", str(run_dir(args.study)),
            "--provider", "custom", "--base-url", f"http://127.0.0.1:{server['port']}/v1",
            "--route", f"{server['model_id']}==", "--reps", str(args.reps), "--first-rep", str(args.first_rep),
            "--model-revision", model, "--context-length", str(server["context_length"]),
            "--episode-timeout", str(EPISODE_TIMEOUT_S), "--episodes", ",".join(EPISODE_ORDER),
            "--learn-first", str(LEARN_FIRST_TRIES)])


def cmd_run(args: argparse.Namespace) -> dict:
    server = _state("server")
    if not server:
        raise SystemExit("start the server first (serve)")
    target, checkpoint = run_dir(args.study), Path(args.checkpoint) / "run"
    target.mkdir(parents=True, exist_ok=True)
    _run(["chown", "-R", f"{AGENT_USER}:{AGENT_USER}", target.parent], PROBE_TIMEOUT_S)
    process = subprocess.Popen(live_command(args, server))
    while process.poll() is None:
        try:
            process.wait(timeout=SYNC_EVERY_S)
        except subprocess.TimeoutExpired:
            try:
                mirror(target, checkpoint)
            except OSError as error:  # Drive can drop a write: the next copy tries again
                print(f"checkpoint copy failed, trying again in {SYNC_EVERY_S} s: {error}", flush=True)
    mirror(target, checkpoint)
    return {"exit_code": process.returncode, "run": str(target), "checkpoint": str(checkpoint)}


def cmd_check_preset(_args: argparse.Namespace) -> dict:
    """Run "preset" as the agent's account, the one with Hermes' environment, and fail if it fails."""
    out = f"/home/{AGENT_USER}/presets.generated.ini"
    process = subprocess.run(as_agent([agent_python(), "evals/learning_loop/gpu_study.py", "preset", "--out", out]),
                             timeout=BUILD_TIMEOUT_S, check=False)
    if process.returncode != 0:
        raise SystemExit("the preset Hermes generates for the downloaded file is not the pre-registered one")
    return {"preset": "matches", "sha256": sha256_of(PINNED_PRESET)}


def cmd_manifest(args: argparse.Namespace) -> dict:
    """Everything a reader needs to check what ran, one file per session (a resumed block can span machines),
    written beside the results and into the checkpoint."""
    commit = _run(["git", "-C", REPO, "rev-parse", "HEAD"], GIT_TIMEOUT_S).stdout.strip()
    platform = {k: os.environ[k] for k in ("COLAB_RELEASE_TAG", "COLAB_GPU", "KAGGLE_KERNEL_RUN_TYPE",
                                            "KAGGLE_DOCKER_IMAGE") if os.environ.get(k)}
    manifest = {"study": args.study, "hermes_commit": commit, "platform": platform, "model": MODEL,
                "engine": _state("engine"), "server": _state("server"), "preset_sha256": sha256_of(PINNED_PRESET),
                "agent_user": AGENT_USER, "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    name = f"study-{manifest['written'].replace(':', '')}.json"
    for target in (run_dir(args.study) / "manifests", Path(args.checkpoint) / "run" / "manifests"):
        target.mkdir(parents=True, exist_ok=True)
        (target / name).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def cmd_sync(args: argparse.Namespace) -> dict:
    return {"copied": mirror(run_dir(args.study), Path(args.checkpoint) / "run")}


COMMANDS = {"engine": cmd_engine, "model": cmd_model, "preset": cmd_preset, "check-preset": cmd_check_preset,
            "serve": cmd_serve, "user": cmd_user, "agent-env": cmd_agent_env, "restore": cmd_restore, "run": cmd_run,
            "manifest": cmd_manifest, "sync": cmd_sync}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--out", default=str(STUDY_ROOT / "presets.generated.ini"), help="preset: where to write it")
    parser.add_argument("--protect", nargs="*", default=[], help="user: paths the agent's account must not read")
    parser.add_argument("--study", default="calibration", help="restore/run: the run's name")
    parser.add_argument("--checkpoint", default=str(STUDY_ROOT / "checkpoint"), help="restore/run: where it is mirrored")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--first-rep", type=int, default=0)
    args = parser.parse_args(argv)
    result = COMMANDS[args.command](args)
    print(json.dumps(result, indent=1, default=str))
    return 0 if result.get("exit_code", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

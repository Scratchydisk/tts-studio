# MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an HTTP-based MCP server that exposes TTS Studio's video captioning pipeline to LLM tools (Claude Desktop, Claude Code) with a job queue, ETA estimation, and smart UI prompting.

**Architecture:** Single Python module (`tts_tests/mcp_server.py`) containing a FastMCP server with 7 tools, an in-memory job queue with a single worker thread, and a job store dict. The server runs as an HTTP service via `tts-studio mcp` CLI subcommand. Configuration is a single `mcp_config.json` file with a `ui_url` field.

**Tech Stack:** Python 3.10+, `mcp[http]` (official MCP Python SDK), existing `tts_tests` modules (api, profiles, registry, caption.pipeline).

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `tts_tests/mcp_server.py` | Create | MCP server, all 7 tool definitions, job queue, worker thread, job store, config loading, validation, ETA calculation |
| `tts_tests/cli.py` | Modify | Add `mcp` subcommand with `--host` and `--port` flags |
| `pyproject.toml` | Modify | Add `mcp` optional dependency group |
| `mcp_config.json` | Create | Default config with `ui_url` |
| `docs/mcp-server.md` | Create | User-facing MCP server documentation |
| `README.md` | Modify | Add paragraph near top linking to MCP docs |

---

### Task 1: Add `mcp` dependency group to pyproject.toml

**Files:**
- Modify: `pyproject.toml:19-32` (optional-dependencies section)

- [ ] **Step 1: Add the mcp optional dependency**

In `pyproject.toml`, add a new extras group after the existing ones. Insert this line after the `piper` line (line 31) and before the `all` line (line 32):

```toml
mcp = ["mcp[http]>=1.0"]
```

The `all` group does NOT need to include `mcp` — it's a separate concern from TTS model dependencies.

- [ ] **Step 2: Install the new dependency**

Run:
```bash
pip install -e ".[mcp]"
```

Expected: installs successfully, `python -c "from mcp.server.fastmcp import FastMCP; print('ok')"` prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add mcp optional dependency group"
```

---

### Task 2: Create mcp_config.json

**Files:**
- Create: `mcp_config.json`

- [ ] **Step 1: Create the config file**

Create `mcp_config.json` in the project root:

```json
{
    "ui_url": "http://localhost:7860"
}
```

- [ ] **Step 2: Commit**

```bash
git add mcp_config.json
git commit -m "feat: add default MCP server config"
```

---

### Task 3: Build the MCP server core — config, job store, worker thread

**Files:**
- Create: `tts_tests/mcp_server.py`

This task builds the infrastructure: config loading, job data model, job store, queue, worker thread, and ETA calculation. Tools are added in the next tasks.

- [ ] **Step 1: Create mcp_server.py with config loading, job model, job store, queue, and worker**

Create `tts_tests/mcp_server.py`:

```python
"""MCP server for TTS Studio — exposes video captioning to LLM tools."""

import json
import logging
import queue
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

from tts_tests.profiles import list_profiles, get_profile, VoiceProfile

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "mcp_config.json"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MCPConfig:
    ui_url: str = "http://localhost:7860"


def load_config() -> MCPConfig:
    """Load MCP config from mcp_config.json, falling back to defaults."""
    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            return MCPConfig(ui_url=data.get("ui_url", MCPConfig.ui_url))
        except Exception as e:
            logger.warning("Failed to read mcp_config.json: %s", e)
    return MCPConfig()


# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    status: str  # queued / running / completed / failed / cancelled
    submitted_at: float
    params: dict[str, Any]
    started_at: float | None = None
    completed_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    stage: str | None = None


# ---------------------------------------------------------------------------
# Job store and queue
# ---------------------------------------------------------------------------

class JobStore:
    """Thread-safe in-memory job store with queue and ETA tracking."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._completed_durations: list[float] = []

    def create(self, params: dict[str, Any]) -> Job:
        """Create a new job and enqueue it."""
        job = Job(
            id=str(uuid.uuid4()),
            status="queued",
            submitted_at=time.time(),
            params=params,
        )
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all_jobs(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.submitted_at)

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in kwargs.items():
                setattr(job, key, value)

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued job. Returns True if cancelled, False otherwise."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != "queued":
                return False
            job.status = "cancelled"
            return True

    def record_completion(self, duration: float) -> None:
        """Record a completed job's duration for ETA calculation."""
        with self._lock:
            self._completed_durations.append(duration)

    def queue_position(self, job_id: str) -> int:
        """Count of queued jobs submitted before this one (0 = next up)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != "queued":
                return 0
            position = 0
            for other in self._jobs.values():
                if (
                    other.id != job_id
                    and other.status == "queued"
                    and other.submitted_at < job.submitted_at
                ):
                    position += 1
            return position

    def estimate_eta(self, job_id: str) -> str:
        """Estimate time until a queued job starts processing."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != "queued":
                return "N/A"
            if not self._completed_durations:
                return "unknown — no completed jobs to estimate from"

            avg_duration = sum(self._completed_durations) / len(self._completed_durations)

            # Count jobs ahead
            position = 0
            for other in self._jobs.values():
                if (
                    other.id != job_id
                    and other.status == "queued"
                    and other.submitted_at < job.submitted_at
                ):
                    position += 1

            # Factor in the currently running job's remaining time
            running_remaining = 0.0
            for other in self._jobs.values():
                if other.status == "running" and other.started_at is not None:
                    elapsed = time.time() - other.started_at
                    running_remaining = max(0.0, avg_duration - elapsed)
                    break

            eta_seconds = running_remaining + position * avg_duration
            if eta_seconds < 60:
                return f"~{int(eta_seconds)}s"
            return f"~{eta_seconds / 60:.1f}m"

    def next_job_id(self) -> str:
        """Block until a job ID is available from the queue."""
        return self._queue.get()


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

def _run_worker(store: JobStore) -> None:
    """Worker loop — processes one job at a time from the queue."""
    from tts_tests import registry
    from tts_tests.api import caption_video

    registry.discover()

    while True:
        job_id = store.next_job_id()
        job = store.get(job_id)
        if job is None or job.status == "cancelled":
            continue

        store.update(job_id, status="running", started_at=time.time(), stage="starting")

        def on_progress(stage: str, current: int, total: int) -> None:
            store.update(job_id, stage=stage)

        try:
            result = caption_video(
                video_path=job.params["video_path"],
                srt_path=job.params.get("srt_path"),
                profile=job.params["profile"],
                output_path=job.params.get("output_path"),
                output_format=job.params.get("output_format", "mkv"),
                on_progress=on_progress,
            )
            duration = time.time() - job.started_at
            store.update(
                job_id,
                status="completed",
                completed_at=time.time(),
                result={
                    "output_path": str(result.output_path),
                    "duration": result.duration,
                    "segments_count": result.segments_count,
                },
                stage=None,
            )
            store.record_completion(duration)
        except Exception as e:
            store.update(
                job_id,
                status="failed",
                completed_at=time.time(),
                error=str(e),
                stage=None,
            )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_ffmpeg() -> str | None:
    """Return an error message if ffmpeg is not on PATH, else None."""
    if shutil.which("ffmpeg") is None:
        return "ffmpeg not found on PATH. Install it on the server before captioning."
    return None


def _validate_caption_request(
    params: dict[str, Any], config: MCPConfig
) -> str | None:
    """Validate a caption request. Returns an error message or None if valid."""
    # Check ffmpeg
    ffmpeg_err = _check_ffmpeg()
    if ffmpeg_err:
        return ffmpeg_err

    # Check profiles exist at all
    profiles = list_profiles()
    if not profiles:
        return (
            f"No voice profiles configured. "
            f"Open the TTS Studio UI at {config.ui_url} to create one."
        )

    # Check named profile exists
    profile_name = params["profile"]
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles.keys()))
        return (
            f"Profile '{profile_name}' not found. "
            f"Available profiles: {available}. "
            f"Open the UI at {config.ui_url} to create or manage profiles."
        )

    # Check model availability
    from tts_tests import registry
    if not registry.list_models():
        registry.discover()

    vp = profiles[profile_name]
    model_id = vp.model_id
    models = {info.model_id: avail for info, avail in registry.list_models()}
    if model_id not in models or not models[model_id]:
        return (
            f"Profile '{profile_name}' uses model '{model_id}' which isn't installed "
            f"or configured as a remote endpoint. "
            f"Open the UI at {config.ui_url} to check model configuration."
        )

    # Check video file
    video_path = Path(params["video_path"])
    if not video_path.exists():
        return f"Video file not found: {params['video_path']}"

    # Check SRT file
    srt_path = params.get("srt_path")
    if srt_path:
        if not Path(srt_path).exists():
            return f"SRT file not found: {srt_path}"
    else:
        auto_srt = video_path.with_suffix(".srt")
        if not auto_srt.exists():
            return (
                f"No SRT file found. Provide one explicitly or place "
                f"'{video_path.stem}.srt' alongside the video."
            )

    return None


# ---------------------------------------------------------------------------
# MCP server and tools
# ---------------------------------------------------------------------------

mcp_app = FastMCP(
    "TTS Studio",
    stateless_http=True,
    json_response=True,
)

_config: MCPConfig | None = None
_store: JobStore | None = None


def _get_config() -> MCPConfig:
    assert _config is not None
    return _config


def _get_store() -> JobStore:
    assert _store is not None
    return _store


@mcp_app.tool()
def caption_video(
    video_path: str,
    profile: str,
    srt_path: str | None = None,
    output_path: str | None = None,
    output_format: str = "mkv",
) -> dict[str, Any]:
    """Submit a video captioning job. Requires a voice profile (create in TTS Studio UI).

    The video is processed through a 4-stage pipeline: parse SRT, generate
    speech, build timeline, render output. Jobs run sequentially — if other
    jobs are ahead, this one queues and you get a position + ETA.

    Args:
        video_path: Absolute path to the video file on the server.
        profile: Voice profile name (must exist in profiles.json).
        srt_path: Path to SRT subtitle file. Auto-detected as {video_stem}.srt if omitted.
        output_path: Output file path. Defaults to {video}_narrated.{format}.
        output_format: Output format — mkv, mp4, or webm. Defaults to mkv.
    """
    config = _get_config()
    store = _get_store()

    params = {
        "video_path": video_path,
        "profile": profile,
        "srt_path": srt_path,
        "output_path": output_path,
        "output_format": output_format,
    }

    error = _validate_caption_request(params, config)
    if error:
        return {"error": error}

    job = store.create(params)
    position = store.queue_position(job.id)
    eta = store.estimate_eta(job.id)

    return {
        "job_id": job.id,
        "status": "queued",
        "queue_position": position,
        "estimated_wait": eta,
    }


@mcp_app.tool()
def get_job_status(job_id: str) -> dict[str, Any]:
    """Check the status of a captioning job.

    Args:
        job_id: The job ID returned by caption_video.
    """
    store = _get_store()
    job = store.get(job_id)
    if job is None:
        return {"error": f"Job not found: {job_id}"}

    result: dict[str, Any] = {
        "job_id": job.id,
        "status": job.status,
        "video_path": job.params["video_path"],
        "profile": job.params["profile"],
    }

    if job.status == "queued":
        result["queue_position"] = store.queue_position(job.id)
        result["estimated_wait"] = store.estimate_eta(job.id)
    elif job.status == "running":
        result["stage"] = job.stage
        result["elapsed"] = f"{time.time() - job.started_at:.0f}s" if job.started_at else "N/A"
    elif job.status == "completed" and job.result:
        result["output_path"] = job.result["output_path"]
        result["duration"] = job.result["duration"]
        result["segments_count"] = job.result["segments_count"]
    elif job.status == "failed":
        result["error"] = job.error
    elif job.status == "cancelled":
        result["message"] = "Job was cancelled."

    return result


@mcp_app.tool()
def list_jobs() -> dict[str, Any]:
    """List all captioning jobs (queued, running, completed, failed, cancelled)."""
    store = _get_store()
    jobs = store.all_jobs()

    job_list = []
    for job in jobs:
        entry: dict[str, Any] = {
            "job_id": job.id,
            "status": job.status,
            "video_path": job.params["video_path"],
            "profile": job.params["profile"],
            "submitted_at": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(job.submitted_at)
            ),
        }
        if job.status == "queued":
            entry["queue_position"] = store.queue_position(job.id)
            entry["estimated_wait"] = store.estimate_eta(job.id)
        job_list.append(entry)

    return {"jobs": job_list, "total": len(job_list)}


@mcp_app.tool()
def cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel a queued captioning job. Running jobs cannot be cancelled.

    Args:
        job_id: The job ID to cancel.
    """
    store = _get_store()
    job = store.get(job_id)
    if job is None:
        return {"error": f"Job not found: {job_id}"}
    if job.status != "queued":
        return {"error": f"Cannot cancel job with status '{job.status}'. Only queued jobs can be cancelled."}

    store.cancel(job_id)
    return {"job_id": job_id, "status": "cancelled", "message": "Job cancelled successfully."}


@mcp_app.tool()
def list_voice_profiles() -> dict[str, Any]:
    """List all available voice profiles for captioning.

    Voice profiles are created in the TTS Studio web UI. Each profile saves a
    TTS model + voice configuration under a friendly name.
    """
    config = _get_config()
    profiles = list_profiles()

    if not profiles:
        return {
            "profiles": [],
            "message": (
                f"No voice profiles configured. "
                f"Create profiles in the TTS Studio UI at {config.ui_url}."
            ),
        }

    profile_list = []
    for name, vp in sorted(profiles.items()):
        profile_list.append({
            "name": name,
            "model_id": vp.model_id,
            "voice": vp.voice,
            "uses_voice_cloning": vp.reference_audio is not None,
        })

    return {"profiles": profile_list}


@mcp_app.tool()
def get_voice_profile(profile_name: str) -> dict[str, Any]:
    """Get full details of a voice profile.

    Args:
        profile_name: Name of the profile to look up.
    """
    config = _get_config()
    try:
        vp = get_profile(profile_name)
    except KeyError:
        profiles = list_profiles()
        available = ", ".join(sorted(profiles.keys())) if profiles else "none"
        return {
            "error": (
                f"Profile '{profile_name}' not found. "
                f"Available profiles: {available}. "
                f"Open the UI at {config.ui_url} to create or manage profiles."
            )
        }

    return {
        "name": profile_name,
        "model_id": vp.model_id,
        "voice": vp.voice,
        "reference_audio": vp.reference_audio,
        "reference_text": vp.reference_text,
        "notes": vp.notes,
    }


@mcp_app.tool()
def open_ui() -> dict[str, Any]:
    """Check if the TTS Studio web UI is running and get its URL.

    Use this when you need to direct the user to the UI for profile creation,
    model configuration, or other tasks that require the web interface.
    """
    config = _get_config()
    try:
        resp = httpx.get(config.ui_url, timeout=5.0)
        if resp.status_code == 200:
            return {
                "status": "running",
                "url": config.ui_url,
                "message": f"TTS Studio UI is running at {config.ui_url}",
            }
    except Exception:
        pass

    return {
        "status": "not_running",
        "url": config.ui_url,
        "message": (
            f"TTS Studio UI is not responding at {config.ui_url}. "
            f"Start it on the server with: ./run.sh"
        ),
    }


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

def run_server(host: str = "0.0.0.0", port: int = 8900) -> None:
    """Start the MCP server."""
    global _config, _store

    _config = load_config()
    _store = JobStore()

    # Start worker thread
    worker = threading.Thread(target=_run_worker, args=(_store,), daemon=True)
    worker.start()

    logger.info("TTS Studio MCP server starting on %s:%d", host, port)
    logger.info("UI URL: %s", _config.ui_url)

    mcp_app.run(transport="streamable-http", host=host, port=port)
```

- [ ] **Step 2: Verify the module imports cleanly**

Run:
```bash
python -c "from tts_tests.mcp_server import mcp_app, run_server; print('ok')"
```

Expected: prints `ok` (no import errors).

- [ ] **Step 3: Commit**

```bash
git add tts_tests/mcp_server.py
git commit -m "feat: add MCP server with job queue, worker, and 7 tools"
```

---

### Task 4: Add `mcp` CLI subcommand

**Files:**
- Modify: `tts_tests/cli.py:207-208` (subparsers section)
- Modify: `tts_tests/cli.py:276-283` (handlers dict)

- [ ] **Step 1: Add the `_cmd_mcp` handler and subparser**

In `tts_tests/cli.py`, add the handler function after `_cmd_profiles` (after line 196):

```python
def _cmd_mcp(args: argparse.Namespace) -> int:
    """Start the MCP server."""
    from tts_tests.mcp_server import run_server

    _suppress_warnings()
    run_server(host=args.host, port=args.port)
    return 0
```

Add the subparser after the `profiles` subparser (after line 262):

```python
    # mcp
    mcp_parser = subparsers.add_parser("mcp", help="Start MCP server")
    mcp_parser.add_argument("--host", default="0.0.0.0")
    mcp_parser.add_argument("--port", type=int, default=8900)
```

Add `"mcp": _cmd_mcp` to the `handlers` dict (line 277):

```python
    handlers = {
        "serve": _cmd_serve,
        "caption": _cmd_caption,
        "batch": _cmd_batch,
        "generate": _cmd_generate,
        "profiles": _cmd_profiles,
        "mcp": _cmd_mcp,
    }
```

- [ ] **Step 2: Verify the subcommand is registered**

Run:
```bash
tts-studio mcp --help
```

Expected: shows help with `--host` and `--port` options.

- [ ] **Step 3: Commit**

```bash
git add tts_tests/cli.py
git commit -m "feat: add 'tts-studio mcp' CLI subcommand"
```

---

### Task 5: Write user-facing documentation

**Files:**
- Create: `docs/mcp-server.md`
- Modify: `README.md` (add paragraph near top)

- [ ] **Step 1: Create docs/mcp-server.md**

```markdown
# MCP Server

TTS Studio includes an MCP (Model Context Protocol) server that lets AI assistants like Claude caption videos using your configured voice profiles. The server exposes the captioning pipeline as tools that Claude can call directly from a conversation.

## Prerequisites

- Voice profiles configured in the TTS Studio UI (the MCP server reads profiles but doesn't create them)
- `ffmpeg` installed on the server
- The `mcp` extras installed: `pip install -e ".[mcp]"`

## Quick start

1. Configure the UI URL in `mcp_config.json`:

```json
{
    "ui_url": "http://your-server:7860"
}
```

2. Start the MCP server:

```bash
tts-studio mcp --host 0.0.0.0 --port 8900
```

3. Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
    "mcpServers": {
        "tts-studio": {
            "url": "http://your-server:8900/mcp"
        }
    }
}
```

For local use, replace `your-server` with `localhost`.

## Configuration

The MCP server reads `mcp_config.json` from the project root:

| Field | Default | Description |
|---|---|---|
| `ui_url` | `http://localhost:7860` | URL where the TTS Studio Gradio UI is running |

This URL is used in error messages to direct users to the UI when action is needed (e.g., creating voice profiles).

## Available tools

### caption_video

Submit a video for captioning. The video is processed through a 4-stage pipeline (parse SRT, generate speech, build timeline, render). Jobs run sequentially via a queue.

**Parameters:**
- `video_path` (required) — absolute path to the video file on the server
- `profile` (required) — voice profile name
- `srt_path` — path to SRT file (auto-detected if omitted)
- `output_path` — output file path (defaults to `{video}_narrated.mkv`)
- `output_format` — mkv, mp4, or webm

**Returns:** job ID, queue position, and estimated wait time.

### get_job_status

Check a captioning job's progress. Returns current pipeline stage while running, output path when completed, or error message if failed.

### list_jobs

View all jobs — queued, running, completed, failed, and cancelled.

### cancel_job

Cancel a queued job. Running jobs cannot be cancelled.

### list_voice_profiles

List all configured voice profiles with their model and voice settings.

### get_voice_profile

Get full details of a specific profile.

### open_ui

Check if the TTS Studio UI is running and get its URL.

## Example workflow

1. Claude calls `list_voice_profiles` to see what voices are available
2. User says "Caption my video at /data/videos/intro.mp4 with the emma voice"
3. Claude calls `caption_video` with `video_path="/data/videos/intro.mp4"` and `profile="emma"`
4. Claude receives job ID and queue position
5. Claude periodically calls `get_job_status` to check progress
6. When complete, Claude reports the output path to the user

If no profiles exist, Claude calls `open_ui` to check if the UI is running and directs the user to create profiles there.

## Job queue

Jobs are processed one at a time to avoid GPU contention. When multiple videos are submitted, they queue up and run sequentially. Each tool response includes:

- **Queue position** — how many jobs are ahead
- **ETA** — estimated time until the job starts (based on rolling average of completed job durations)

The queue is in-memory only — if the server restarts, pending jobs are lost and must be resubmitted.
```

- [ ] **Step 2: Add MCP paragraph to README.md**

Add a new section after the "Quick start" section's CLI block (after line 33, before "## Supported models"). Insert:

```markdown
## MCP server

TTS Studio includes an [MCP server](docs/mcp-server.md) that lets AI assistants like Claude caption videos using your voice profiles. Start it with `tts-studio mcp` and connect from Claude Desktop or Claude Code. See the [MCP server docs](docs/mcp-server.md) for setup instructions.
```

- [ ] **Step 3: Commit**

```bash
git add docs/mcp-server.md README.md
git commit -m "docs: add MCP server documentation and README link"
```

---

### Task 6: Manual smoke test

- [ ] **Step 1: Start the MCP server**

Run in a terminal:
```bash
tts-studio mcp --port 8900
```

Expected: server starts, logs show "TTS Studio MCP server starting on 0.0.0.0:8900".

- [ ] **Step 2: Test the streamable HTTP endpoint is responding**

In another terminal:
```bash
curl -s http://localhost:8900/ | head -20
```

Expected: some response (not connection refused). The MCP protocol requires proper MCP client communication, but the HTTP server itself should be up.

- [ ] **Step 3: Test with a quick Python MCP client snippet**

```bash
python -c "
import httpx
# Test that the server is accepting HTTP connections
resp = httpx.get('http://localhost:8900/', timeout=5)
print(f'Status: {resp.status_code}')
"
```

Expected: returns a status code (likely 200 or 405), confirming the server is listening.

- [ ] **Step 4: Stop the server and verify clean shutdown**

Ctrl+C the server process. Expected: exits cleanly without errors.

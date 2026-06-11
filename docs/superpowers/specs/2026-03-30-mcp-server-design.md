# TTS Studio MCP Server — Design Spec

## Overview

An MCP (Model Context Protocol) server that exposes TTS Studio's video captioning pipeline to LLM-powered tools like Claude Desktop and Claude Code. The server runs as an HTTP service on the same machine as TTS Studio (typically a remote GPU server), and MCP clients connect to it over the network.

The MCP server is a **consumer** of voice profiles — profile creation and management stays in the Gradio UI. The server intelligently detects when profiles are missing or the UI isn't running, and guides the user to fix it.

## Architecture

### New files

- `tts_tests/mcp_server.py` — MCP server, tool definitions, job queue, worker thread
- `mcp_config.json` — server configuration (single `ui_url` field)
- `docs/mcp-server.md` — user-facing documentation page

### Modified files

- `tts_tests/cli.py` — new `mcp` subcommand
- `pyproject.toml` — new `mcp` optional dependency group
- `README.md` — paragraph near the top linking to the MCP docs page

### Dependencies

- `mcp[http]` — the official Python MCP SDK with HTTP/SSE transport support. Added as an optional dependency group (`pip install -e ".[mcp]"`).

### Process model

Single Python process running an HTTP/SSE MCP server. Inside the process:

- **MCP server** — handles tool calls from connected clients
- **Job queue** — `queue.Queue` holding pending caption requests
- **Worker thread** — single daemon thread that pulls jobs one at a time and calls `tts_tests.api.caption_video()` directly
- **Job store** — plain `dict` mapping job IDs to job records, tracking state and results

No external services, no database, no Node.js. The server imports `tts_tests` modules directly.

### Deployment

On the GPU server:

```bash
tts-studio mcp --host 0.0.0.0 --port 8900
```

In the user's Claude Desktop MCP config:

```json
{
  "mcpServers": {
    "tts-studio": {
      "url": "http://gpu-server:8900/mcp"
    }
  }
}
```

For local use, the URL is simply `http://localhost:8900/mcp`.

### Configuration

`mcp_config.json` in the project root:

```json
{
  "ui_url": "http://localhost:7860"
}
```

Single field — the URL where the Gradio UI is running. Used in error messages and `open_ui` tool responses so Claude can direct the user to the right place.

## Tools

### `caption_video` — Submit a captioning job

**Inputs:**
| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `video_path` | string | yes | — | Absolute path to the video file on the server |
| `srt_path` | string | no | auto-detected | Path to SRT subtitle file |
| `profile` | string | yes | — | Voice profile name |
| `output_path` | string | no | `{video}_narrated.{fmt}` | Output file path |
| `output_format` | string | no | `"mkv"` | Output format: mkv, mp4, webm |

**Upfront validation (before queuing):**
1. Profile exists in `profiles.json`
2. Profile's model is available (installed locally or has a remote endpoint)
3. `ffmpeg` is on PATH
4. Video file exists at the given path
5. SRT file exists (at given path, or auto-discoverable as `{video_stem}.srt`)

**Returns on success:** job ID, queue position (0 = will run immediately), estimated time until job starts.

**Returns on validation failure:** error message with actionable guidance, including the UI URL when the fix requires the UI (see Smart UI Prompting section).

### `get_job_status` — Check a job's status

**Input:** `job_id` (string)

**Returns based on status:**
- **queued**: queue position, estimated time until start
- **running**: current pipeline stage (parse/tts/timeline/render), elapsed time
- **completed**: output path, duration, segment count
- **failed**: error message
- **cancelled**: cancellation confirmation

### `list_jobs` — View all jobs

**No inputs.**

**Returns:** all jobs ordered by submission time, each with: job ID, status, video path, profile, submitted timestamp. Queued jobs include position and ETA.

### `cancel_job` — Cancel a queued job

**Input:** `job_id` (string)

**Behavior:** removes queued jobs from the queue and marks them cancelled. Running jobs cannot be cancelled — they complete normally. Returns error if the job is already completed/failed/cancelled.

### `list_profiles` — List available voice profiles

**No inputs.**

**Returns:** all profile names with summary info: model ID, voice name, whether it uses voice cloning (has reference audio). If no profiles exist, includes a message directing to the UI.

### `get_profile` — Get profile details

**Input:** `profile_name` (string)

**Returns:** full profile configuration — model_id, voice, reference_audio path, reference_text, notes.

### `open_ui` — Get UI access info

**No inputs.**

**Behavior:** sends an HTTP GET to the configured `ui_url` to check if the Gradio server is responding.

**Returns:**
- If reachable: "TTS Studio UI is running at {ui_url}"
- If not reachable: "TTS Studio UI is not responding at {ui_url}. Start it on the server with: `./run.sh`"

(Systemctl instructions will be added to docs once service configuration is set up.)

## Job Queue

### Job lifecycle

```
submitted → queued → running → completed
                           ↘ failed
              ↘ cancelled
```

### Job record fields

| Field | Type | Description |
|---|---|---|
| `id` | string (UUID) | Unique job identifier |
| `status` | enum | queued / running / completed / failed / cancelled |
| `submitted_at` | float | Submission timestamp |
| `started_at` | float or None | When the worker picked it up |
| `completed_at` | float or None | When it finished (success or failure) |
| `params` | dict | The caption_video arguments |
| `result` | dict or None | Output path, duration, segment count |
| `error` | string or None | Error message on failure |
| `stage` | string or None | Current pipeline stage while running |

### Queue position and ETA

Reported on submission (`caption_video` response) and on status checks (`get_job_status`):

- **Queue position**: count of jobs ahead in the queue. 0 means the job will run next (or immediately if the worker is idle).
- **ETA calculation**: rolling average of completed job durations in the current session.
  - No completed jobs yet → "unknown — no completed jobs to estimate from"
  - Running job factored in: if a job has been running for 30s and average completion time is 60s, the next queued job's ETA starts at ~30s, plus `(position - 1) * average` for jobs further back.

### Worker thread

- Single daemon thread, started when the MCP server boots.
- Blocks on `queue.Queue.get()`, processes one job at a time.
- Uses the `on_progress` callback from `caption_video()` to update the job's `stage` field in real time.
- Catches all exceptions and marks the job as failed with the error message.
- Jobs are **not persisted** — if the server restarts, the queue is empty. Users resubmit.

## Smart UI Prompting

The MCP server includes the configured `ui_url` in error messages so Claude can relay it directly to the user.

### On `caption_video` validation failure

| Condition | Message |
|---|---|
| No profiles exist | "No voice profiles configured. Open the TTS Studio UI at {ui_url} to create one." |
| Named profile not found | "Profile '{name}' not found. Available profiles: {list}. Open the UI at {ui_url} to create or manage profiles." |
| Profile's model unavailable | "Profile '{name}' uses model '{model_id}' which isn't installed or configured as a remote endpoint. Open the UI at {ui_url} to check model configuration." |
| ffmpeg missing | "ffmpeg not found on PATH. Install it on the server before captioning." |
| Video not found | "Video file not found: {path}" |
| SRT not found | "No SRT file found. Provide one explicitly or place '{stem}.srt' alongside the video." |

### On `list_profiles` returning empty

Returns empty list plus: "No voice profiles configured. Create profiles in the TTS Studio UI at {ui_url}."

### On `open_ui`

- Reachable: "TTS Studio UI is running at {ui_url}"
- Not reachable: "TTS Studio UI is not responding at {ui_url}. Start it on the server with: `./run.sh`"

## CLI

New subcommand added to `tts_tests/cli.py`:

```
tts-studio mcp [--host HOST] [--port PORT]
```

| Flag | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8900` | Port number |

## Documentation

### New page: `docs/mcp-server.md`

User-facing documentation covering:
- What the MCP server does
- Prerequisites (profiles configured in UI, ffmpeg installed)
- How to start the server
- How to configure `mcp_config.json`
- How to connect from Claude Desktop / Claude Code
- Available tools and what they do
- Example workflow

### README update

A paragraph near the top of `README.md` introducing the MCP server with a link to `docs/mcp-server.md`.

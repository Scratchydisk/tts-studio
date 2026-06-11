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

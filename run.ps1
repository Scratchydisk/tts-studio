#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# uv provisions Python 3.12 without admin rights and resolves dependencies
# in seconds, failing fast with a clear message on conflicting pins where
# pip backtracks for hours.
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host 'Error: uv is not installed. Install it with:' -ForegroundColor Red
    Write-Host '  irm https://astral.sh/uv/install.ps1 | iex'
    Write-Host 'See https://docs.astral.sh/uv/getting-started/installation/'
    exit 1
}

# Use a separate venv dir so we don't clobber a Linux venv created by run.sh.
# Recreate if missing, relocated, or on Python < 3.11 (f5-tts caps numpy at
# 1.26.4 on <3.11, conflicting with dia's numpy>=2.2.4).
$venvDir = Join-Path $PSScriptRoot 'venv-win'
$venvCfg = Join-Path $venvDir 'pyvenv.cfg'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'
$venvOk = (Test-Path $venvCfg) -and (Select-String -Path $venvCfg -SimpleMatch $venvDir -Quiet)
if ($venvOk) {
    & $venvPython -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Existing venv uses Python < 3.11; recreating...'
        $venvOk = $false
    }
}
if (-not $venvOk) {
    Write-Host 'Creating virtual environment with Python 3.12...'
    if (Test-Path $venvDir) { Remove-Item -Recurse -Force $venvDir }
    uv venv $venvDir --python 3.12
}

. (Join-Path $venvDir 'Scripts\Activate.ps1')

$extras = 'all,mcp'

# Pick the PyTorch wheel index that matches this machine's hardware. The default
# PyPI wheel (cu126) covers Maxwell..Hopper but has no Blackwell (sm_120) kernels,
# so a 50-series GPU silently fails at runtime; the cu128 wheel adds sm_120 but
# drops Maxwell. We only override the default in the two cases it gets wrong:
# Blackwell (needs cu128) and no NVIDIA GPU (lean CPU wheel). Empty = keep default.
function Get-TorchIndex {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $cap = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
        if ($cap -and ($cap.Trim() -match '^(\d+)') -and [int]$Matches[1] -ge 12) {
            return 'https://download.pytorch.org/whl/cu128'  # Blackwell (sm_120+)
        }
        # GPU present but pre-Blackwell (or cap unknown): default wheel covers it
        return ''
    }
    return 'https://download.pytorch.org/whl/cpu'  # no NVIDIA GPU
}

if (-not (Get-Command tts-studio -ErrorAction SilentlyContinue)) {
    Write-Host "Installing dependencies ($extras)..."
    uv pip install -e ".[$extras]"
    # uv is a native exe, so $ErrorActionPreference doesn't catch its failures
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Dependency install failed (exit code $LASTEXITCODE); not starting servers."
    }
    # The resolve above pulls the default PyPI torch (cu126) because the model
    # extras depend on torch transitively. Re-pin torch to the wheel that matches
    # this hardware AFTER the full install, so the resolver can't downgrade it
    # back. Empty index = the default wheel is already correct for this machine.
    $torchIndex = Get-TorchIndex
    if ($torchIndex) {
        Write-Host "Re-pinning PyTorch for this hardware from $torchIndex ..."
        uv pip install --index-url $torchIndex --upgrade torch torchaudio
        if ($LASTEXITCODE -ne 0) {
            Write-Error "PyTorch re-pin failed (exit code $LASTEXITCODE); not starting servers."
        }
    }
} else {
    Write-Host 'Dependencies already installed.'
}

Write-Host 'Starting MCP server on port 8900...'
$mcp = Start-Process tts-studio -ArgumentList 'mcp' -NoNewWindow -PassThru

Write-Host 'Starting TTS Studio UI...'
# Open browser once the server is actually ready (timeout after 60s)
Start-Job -ScriptBlock {
    foreach ($i in 1..60) {
        try {
            Invoke-WebRequest -Uri 'http://localhost:7860' -UseBasicParsing -TimeoutSec 2 | Out-Null
            break
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    Start-Process 'http://localhost:7860'
} | Out-Null

# Run UI in foreground; kill MCP server on exit
try {
    tts-studio
} finally {
    if ($mcp -and -not $mcp.HasExited) {
        Stop-Process -Id $mcp.Id -Force -ErrorAction SilentlyContinue
    }
}

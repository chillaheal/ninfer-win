# start-claude-ninfer.ps1 - launch Claude Code pointed at a local ninfer-serve.
#
# The server must already be running (GUI Serve button, or ninfer-serve.exe from
# dist/). This script only sets the env vars Claude Code needs and starts claude;
# it never starts or stops the server itself. No proxy or adapter is required -
# ninfer-serve speaks the Anthropic Messages protocol natively.
#
# Context window: when the serve reports its context accounting in /health
# ({"context": {"max_context": N, "kv_capacity": M}} - all builds from
# 2026-09-02), the script sizes Claude Code's window to the serve's
# single-request limit (min of the two) minus a small margin, so compaction
# can never push a request past what the engine accepts. Set via env vars
# (CLAUDE_CODE_MAX_CONTEXT_TOKENS + CLAUDE_CODE_AUTO_COMPACT_WINDOW), which take
# precedence over settings.json. Old serve builds without the field launch
# unchanged (a warning is printed).
#
# Usage:
#   .\start-claude-ninfer.ps1                    # interactive session
#   .\start-claude-ninfer.ps1 -p "Say OK"        # one-shot (any claude args pass through)
#
# Always launches with --dangerously-skip-permissions (no permission prompts).
#
# Non-default server port? Edit $Port below. (No named parameters on purpose:
# PowerShell would prefix-match e.g. claude's -p onto a -Port param.)

$Port = 8888
$base = "http://127.0.0.1:$Port"

# Warn (and stop) if nothing is listening - claude would just fail to connect.
try {
    $r = Invoke-WebRequest -Uri "$base/health" -UseBasicParsing -TimeoutSec 3
    if ($r.StatusCode -ne 200) { throw "health returned $($r.StatusCode)" }
} catch {
    Write-Warning ("No ninfer-serve answering at {0} - start the server first (GUI Serve button, or .\dist\ninfer-serve.exe) and run this script again." -f $base)
    exit 1
}

# Auth is disabled on the server unless --api-key was passed, so the token value is
# arbitrary. We use a bearer ANTHROPIC_AUTH_TOKEN (NOT ANTHROPIC_API_KEY): a non-Anthropic
# API key triggers Claude Code's "trust this custom key?" prompt and, once declined, gets
# recorded in .claude.json under customApiKeyResponses.rejected -- after which Claude refuses
# the key and drops to the OAuth "log in" screen. A bearer AUTH_TOKEN skips that gate entirely
# (same mechanism as the working unsloth setup). This also makes the script self-contained: it
# no longer depends on an inherited token being present in the launching terminal.
if ($env:ANTHROPIC_API_KEY) { Remove-Item Env:\ANTHROPIC_API_KEY }   # don't let a stale/rejected key win
$env:ANTHROPIC_BASE_URL   = $base   # no /v1 suffix - Claude Code appends /v1/messages itself
$env:ANTHROPIC_AUTH_TOKEN = "sk-ninfer-local"

# ANTHROPIC_MODEL is intentionally left alone: the server accepts any model string,
# so an existing config (e.g. unsloth/Qwen3.8-27B-GGUF) works unchanged.

# Size Claude Code's context window to the serve's single-request limit minus a margin.
# The serve rejects prompts past max_context, and kv_capacity is the token pool it
# reserved (always >= max_context), so the binding limit is the smaller of the two.
# The margin keeps compaction output + system prompt inside the window with headroom.
# Both env vars take precedence over settings.json (the 115000 autoCompactWindow there
# is the fallback for old serve builds that do not report the context field).
$Margin = 5000
$health = $null
try { $health = $r.Content | ConvertFrom-Json } catch { }
if ($health -and $health.context -and
    $health.context.max_context -and $health.context.kv_capacity) {
    $limit = [math]::Min([int64]$health.context.max_context,
                         [int64]$health.context.kv_capacity) - $Margin
    if ($limit -lt 16384) {
        Write-Warning ("serve context limit {0} is below the minimum usable window - leaving Claude Code's window untouched." -f
            ([math]::Min([int64]$health.context.max_context, [int64]$health.context.kv_capacity)))
    } else {
        $env:CLAUDE_CODE_MAX_CONTEXT_TOKENS  = "$limit"
        $env:CLAUDE_CODE_AUTO_COMPACT_WINDOW = "$limit"
        Write-Output ("serve context: max_context={0} kv_capacity={1} -> Claude window {2} (margin {3})" -f
            $health.context.max_context, $health.context.kv_capacity, $limit, $Margin)
    }
} else {
    Write-Warning "serve /health has no context field (old serve build?) - Claude Code window left at the settings.json value."
}

# --dangerously-skip-permissions: runs without permission prompts (YOLO mode).
# Intended for this local, trusted setup against the local ninfer-serve.
& claude --dangerously-skip-permissions @args
exit $LASTEXITCODE

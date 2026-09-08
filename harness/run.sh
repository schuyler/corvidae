#!/usr/bin/env bash
# Buster harness — remote-host entry point. The single SSH command's target
# (harness/run-remote.sh invokes this via ssh). Ensures services are up,
# snapshots state, runs the probe driver, and writes a self-contained
# run-report directory. No mid-run cross-network puppeteering.
#
# llama-server is a standing, Schuyler-managed service at 127.0.0.1:8080 —
# this script only checks reachability and snapshots its reported config.
# It never launches, stops, or restarts llama-server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/config.local.sh" ] && source "$SCRIPT_DIR/config.local.sh"
BUSTER_ROOT="${BUSTER_ROOT:-$HOME/code/llm/buster}"
REPO_DIR="$BUSTER_ROOT/repo"
STATE_DIR="$BUSTER_ROOT/state"
RUN_DIR="$BUSTER_ROOT/runs/$(date -u +%Y%m%dT%H%M%SZ)"
LLAMA_URL="http://127.0.0.1:8080"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# uv installs to ~/.local/bin, which non-interactive ssh sessions don't put
# on PATH (no login shell, no .bashrc sourcing).
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$RUN_DIR" "$STATE_DIR" "$BUSTER_ROOT/snapshots"
cd "$REPO_DIR"
GIT_REV="$(git rev-parse HEAD)"

abort() {
    # Writes a self-contained report.json/report.txt for a run that never
    # reached the driver, then exits nonzero. No silent empty run dirs.
    local reason="$1"
    REASON="$reason" RUN_DIR="$RUN_DIR" GIT_REV="$GIT_REV" LLAMA_URL="$LLAMA_URL" \
        STARTED_AT="$STARTED_AT" python3 - <<'PYEOF'
import json, os, time

report = {
    "run": {
        "started_at": os.environ["STARTED_AT"],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_rev": os.environ["GIT_REV"],
        "llama_endpoint": os.environ["LLAMA_URL"],
        "aborted": True,
        "abort_reason": os.environ["REASON"],
    },
    "probes": [],
}
run_dir = os.environ["RUN_DIR"]
with open(os.path.join(run_dir, "report.json"), "w") as f:
    json.dump(report, f, indent=2)
with open(os.path.join(run_dir, "report.txt"), "w") as f:
    f.write(f"ABORTED: {os.environ['REASON']}\n")
PYEOF
    echo "ABORTED: $reason" >&2
    echo "Report: $RUN_DIR/report.json" >&2
    exit 1
}

# --- 1. ngircd reachability — system service, Schuyler-managed. The harness
#        never starts, stops, or configures it, same contract as llama-server.
if ! (exec 3<>/dev/tcp/127.0.0.1/6667) 2>/dev/null; then
    abort "ngircd unreachable at 127.0.0.1:6667 (Schuyler-managed system service — is it running?)"
fi

# --- 2. llama-server reachability (never launched/stopped by this script) --
if ! curl -sf "$LLAMA_URL/health" >/dev/null; then
    abort "llama-server unreachable at $LLAMA_URL/health (Schuyler-managed service — is it running?)"
fi

# --- 3. stop any previous buster-daemon, snapshot state ---------------------
screen -S buster-daemon -X quit >/dev/null 2>&1 || true
sleep 1
"$REPO_DIR/harness/snapshot.sh"

# Probe channels are fixed names (#p-interleave etc.) and sessions.db is
# never truncated on its own — without this, conversation history from
# every prior run stays loaded, so probes see planted tokens/answers from
# past runs before the current exchange ever resolves. Reset once here,
# after the snapshot backed up the prior state; daemon restarts *within*
# this run (restart_recovery) still read the same fresh sessions.db.
rm -f "$STATE_DIR/sessions.db" "$STATE_DIR/sessions.db-wal" "$STATE_DIR/sessions.db-shm"

# The daemon logs to $STATE_DIR/corvidae.log per its own config, not to
# daemon.log (screen's stdout/stderr capture) — reset the rotated siblings
# too so this run's copy into $RUN_DIR isn't prior runs' content. Prior
# state is retained by the step-3 snapshot, not by these files.
rm -f "$STATE_DIR"/corvidae.log*

# --- 4. render config --------------------------------------------------------
sed "s|@STATE_DIR@|$STATE_DIR|g" "$REPO_DIR/harness/buster.yaml.in" > "$STATE_DIR/agent.yaml"
CONFIG_SHA="$(sha256sum "$STATE_DIR/agent.yaml" | awk '{print $1}')"

# --- 5. record serving dimensions from the live server, verbatim -----------
curl -sf "$LLAMA_URL/props" > "$RUN_DIR/server_props.json" \
    || echo '{"error":"GET /props failed"}' > "$RUN_DIR/server_props.json"
curl -sf "$LLAMA_URL/v1/models" > "$RUN_DIR/server_models.json" \
    || echo '{"error":"GET /v1/models failed"}' > "$RUN_DIR/server_models.json"

# --- 6. warm the uv environment, then start buster-daemon ------------------
# First provisioning on a fresh checkout (interpreter + dependency fetch)
# takes ~21s observed — do it here, synchronously, so that cost lands
# before the daemon-startup/driver-join windows instead of eating into
# them from inside the screen session.
uv sync --project "$REPO_DIR"

screen -dmS buster-daemon bash -c "
    cd '$REPO_DIR' &&
    exec uv run corvidae serve --config '$STATE_DIR/agent.yaml' >> '$RUN_DIR/daemon.log' 2>&1
"

# --- 7. run the probe driver (it waits for Buster's IRC join itself) -------
# driver.py is stdlib-only — run under the system interpreter, not corvidae's
# uv-managed venv (no dependency sync needed just to run the driver).
set +e
python3 "$REPO_DIR/harness/driver.py" \
    --run-dir "$RUN_DIR" \
    --state-dir "$STATE_DIR" \
    --repo-dir "$REPO_DIR" \
    --started-at "$STARTED_AT" \
    --git-rev "$GIT_REV" \
    --config-sha256 "$CONFIG_SHA" \
    --server-props-file "$RUN_DIR/server_props.json" \
    --server-models-file "$RUN_DIR/server_models.json"
DRIVER_EXIT=$?
set -e

# --- 8. daemon liveness after the driver finished; stop it; finalize report -
if screen -list 2>/dev/null | grep -q "buster-daemon"; then
    DAEMON_CRASHED=false
else
    DAEMON_CRASHED=true
fi
screen -S buster-daemon -X quit >/dev/null 2>&1 || true
sleep 1

# state/corvidae.log is where the daemon actually logs (daemon.log only
# catches pre-logging screen stdout/stderr); copy it in so the run dir is
# self-contained. The sleep above mirrors step 3's, for the same reason:
# screen -X quit signals asynchronously and returns immediately, so without
# it the copy can race the daemon's shutdown records.
cp "$STATE_DIR"/corvidae.log* "$RUN_DIR"/ 2>/dev/null || true

FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

RUN_DIR="$RUN_DIR" DAEMON_CRASHED="$DAEMON_CRASHED" FINISHED_AT="$FINISHED_AT" python3 - <<'PYEOF'
import json, os

run_dir = os.environ["RUN_DIR"]
path = os.path.join(run_dir, "report.json")
with open(path) as f:
    report = json.load(f)
report["run"]["daemon_crashed"] = os.environ["DAEMON_CRASHED"] == "true"
report["run"]["finished_at"] = os.environ["FINISHED_AT"]
with open(path, "w") as f:
    json.dump(report, f, indent=2)

txt_path = os.path.join(run_dir, "report.txt")
with open(txt_path) as f:
    text = f.read()
text = text.replace("PENDING_DAEMON_CRASHED", os.environ["DAEMON_CRASHED"])
text = text.replace("PENDING_FINISHED_AT", os.environ["FINISHED_AT"])
with open(txt_path, "w") as f:
    f.write(text)
PYEOF

cat "$RUN_DIR/report.txt"
echo "Report: $RUN_DIR/report.json"

if [ "$DRIVER_EXIT" -ne 0 ] || [ "$DAEMON_CRASHED" = "true" ]; then
    exit 1
fi
exit 0

#!/usr/bin/env bash
# =============================================================================
# cron_signals_jp.sh — Weekly Numerai Signals JP submission wrapper
# =============================================================================
# Schedule: Every Saturday 10:00 JST (01:00 UTC)
# Usage:    ./scripts/cron_signals_jp.sh
#           crontab -e → see scripts/cron_signals_jp.crontab
# =============================================================================
set -euo pipefail

# --- Constants ---------------------------------------------------------------
PROJECT_ROOT="/Users/MBP/Desktop/Project_Asset_Shield"
LOG_DIR="${PROJECT_ROOT}/data/numerai_signals_jp"
LOG_FILE="${LOG_DIR}/cron.log"
SCRIPT="${PROJECT_ROOT}/scripts/numerai_signals_jp.py"
MAX_LOG_LINES=5000

# --- PATH setup (macOS: cron has minimal PATH) ------------------------------
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${HOME}/.local/bin:${PATH}"

# --- Functions ---------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" | tee -a "${LOG_FILE}"
}

die() {
    log "ERROR: $*"
    exit 1
}

# --- Pre-flight checks -------------------------------------------------------
cd "${PROJECT_ROOT}" || die "Cannot cd to ${PROJECT_ROOT}"

mkdir -p "${LOG_DIR}"

# Load .env if present (NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY, JQUANTS_*, etc.)
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
else
    log "WARN: .env not found — assuming env vars are already set"
fi

# Verify python3 is available
PYTHON="$(command -v python3 2>/dev/null)" || die "python3 not found in PATH"

# Verify the script exists
[[ -f "${SCRIPT}" ]] || die "Script not found: ${SCRIPT}"

# --- Rotate log if too large -------------------------------------------------
if [[ -f "${LOG_FILE}" ]]; then
    line_count=$(wc -l < "${LOG_FILE}" | tr -d ' ')
    if (( line_count > MAX_LOG_LINES )); then
        tail -n $(( MAX_LOG_LINES / 2 )) "${LOG_FILE}" > "${LOG_FILE}.tmp"
        mv "${LOG_FILE}.tmp" "${LOG_FILE}"
        log "Log rotated (was ${line_count} lines)"
    fi
fi

# --- Run ---------------------------------------------------------------------
log "========== Numerai Signals JP: START =========="
log "Python: ${PYTHON} ($(${PYTHON} --version 2>&1))"
log "Command: ${PYTHON} ${SCRIPT} --mode full"

start_ts=$(date +%s)

if "${PYTHON}" "${SCRIPT}" --mode full >> "${LOG_FILE}" 2>&1; then
    elapsed=$(( $(date +%s) - start_ts ))
    log "========== Numerai Signals JP: SUCCESS (${elapsed}s) =========="
    exit 0
else
    rc=$?
    elapsed=$(( $(date +%s) - start_ts ))
    log "========== Numerai Signals JP: FAILED rc=${rc} (${elapsed}s) =========="
    exit ${rc}
fi

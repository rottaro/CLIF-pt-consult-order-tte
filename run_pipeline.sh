#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════════
#  run_pipeline.sh — Execute the full CLIF-pt-consult-order-tte pipeline
#
#  Steps:
#    1. Python  01_cohort.ipynb -> Python
#    2. Python  02_data_gathering.ipynb -> Python
#    3. Python  03_calculations.ipynb -> Python
#    4. R       04_CCW.R
#
#  Usage:  bash run_pipeline.sh
# ════════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── colours ──────────────────────────────────────────────────────────────────
GREEN="\033[32m"; RED="\033[31m"; CYAN="\033[36m"; YELLOW="\033[33m"
BOLD="\033[1m"; RESET="\033[0m"

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}"
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# ── logging ──────────────────────────────────────────────────────────────────
log() { echo -e "$1" | tee -a "$LOG_FILE"; }

log "${CYAN}${BOLD} CLIF PT Consult-order-tte Pipeline${RESET}"
log "Started: $(date)"
log "Log: ${LOG_FILE}"
log ""

# ── environment (uv) ─────────────────────────────────────────────────────────
REQUIREMENTS_FILE="${1:-requirements.txt}"
ENV_NAME="${2:-cpttenv}"
 
# ── Checks ────────────────────────────────────────────────────
 
if ! command -v python3 &>/dev/null; then
  log " python3 not found. Please install Python 3 first."
  return 1 2>/dev/null || exit 1
fi
 
if [ ! -f "$REQUIREMENTS_FILE" ]; then
  log "XX  Requirements file '$REQUIREMENTS_FILE' not found."
  log "    Usage: source setup_env.sh [requirements_file] [env_name]"
  return 1 2>/dev/null || exit 1
fi
 
# ── Create virtual environment ────────────────────────────────
 
if [ -d "$ENV_NAME" ]; then
  log "!!   Virtual environment '$ENV_NAME' already exists — skipping creation."
else
  log "   Creating virtual environment '$ENV_NAME'..."
  python3 -m venv "$ENV_NAME"
  if [ $? -ne 0 ]; then
    log "XX  Failed to create virtual environment."
    return 1 2>/dev/null || exit 1
  fi
  log "   Virtual environment created."
fi
 
# ── Activate ──────────────────────────────────────────────────
 
log "   Activating '$ENV_NAME'..."
 
# Detect OS and pick the right activation script
if [ -f "$ENV_NAME/bin/activate" ]; then
  source "$ENV_NAME/bin/activate"          # macOS / Linux
elif [ -f "$ENV_NAME/Scripts/activate" ]; then
  source "$ENV_NAME/Scripts/activate"      # Windows (Git Bash / WSL)
else
  log "   Could not find activation script in '$ENV_NAME'."
  return 1 2>/dev/null || exit 1
fi
 
log "   Environment activated: $(which python)"
 
# ── Install dependencies ──────────────────────────────────────
 
log "   Installing packages from '$REQUIREMENTS_FILE'..."
pip install -r "$REQUIREMENTS_FILE"
 
if [ $? -eq 0 ]; then
  log ""
  log "   All done! Environment '$ENV_NAME' is active and ready."
  log "    Python : $(python --version)"
  log "    Pip    : $(pip --version)"
  log ""
  log "    To deactivate later, run: deactivate"
else
  log "   Some packages failed to install. Check the output above."
  return 1 2>/dev/null || exit 1
fi

# ── pipeline (cwd = code/ so relative paths work) ───────────────────────────
cd "${PROJECT_ROOT}/code"

# Python steps
log "========== STARTING STEP 1: COHORT =========="
python3 1_cohort.py
log "========== STARTING STEP 2: DATA GATHERING =========="
python3 2_data_gathering.py
log "========== STARTING STEP 3: DATA GATHERING =========="
python3 3_calculations.py

# R steps


log ""
log "Output files in ${PROJECT_ROOT}/output/final/:"
if [ -d "${PROJECT_ROOT}/output/final" ]; then
  # List generated files with sizes
  find "${PROJECT_ROOT}/output/final" -type f -newer "${LOG_FILE}" -exec ls -lh {} \; 2>/dev/null | \
    awk '{printf "  %-8s %s\n", $5, $NF}' | tee -a "$LOG_FILE" || true
  # If nothing newer, just list everything
  FILE_COUNT=$(find "${PROJECT_ROOT}/output/final" -type f | wc -l | tr -d ' ')
  log "  Total files: ${FILE_COUNT}"
else
  log "  (directory not yet created)"
fi

log ""
log "Full log: ${LOG_FILE}"
log "Finished: $(date)"
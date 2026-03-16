#!/bin/bash
# hormuz-ds 定时运行：fetch → compute → report
# 由 LaunchAgent com.hormuz-ds.run 调用，每 4 小时一次

set -euo pipefail

PROJECT_DIR="/Users/xiaohei/Projects/hormuz-ds"
VENV_HORMUZ="${PROJECT_DIR}/.venv/bin/hormuz"
LOG_DIR="${PROJECT_DIR}/data/logs"

mkdir -p "$LOG_DIR"

# 代理
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890}"

cd "$PROJECT_DIR"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') hormuz run ===" >> "${LOG_DIR}/run.log"
"$VENV_HORMUZ" run >> "${LOG_DIR}/run.log" 2>&1
echo "=== done ===" >> "${LOG_DIR}/run.log"

# Auto-push dashboard to GitHub Pages (best-effort)
cd "$PROJECT_DIR"
if git diff --quiet docs/index.html 2>/dev/null; then
    echo "Dashboard unchanged, skip push" >> "${LOG_DIR}/run.log"
else
    git add docs/index.html data/status.html
    git commit -m "auto: dashboard $(date '+%m-%d %H:%M')" --no-gpg-sign >> "${LOG_DIR}/run.log" 2>&1
    git push origin main >> "${LOG_DIR}/run.log" 2>&1 || true
fi

# Sync status to octopus (best-effort, don't block on failure)
"${PROJECT_DIR}/scripts/sync_octopus.sh" >> "${LOG_DIR}/run.log" 2>&1 || true

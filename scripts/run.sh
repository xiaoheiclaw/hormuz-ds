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

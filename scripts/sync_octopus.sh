#!/bin/bash
# 每次 pipeline 跑完后调用，追加一行到 octopus STATUS.md 并推送
# 由 run.sh 在 hormuz run 之后调用

set -euo pipefail

PROJECT_DIR="/Users/xiaohei/Projects/hormuz-ds"
OCTOPUS_DIR="/tmp/octopus-sync"
STATUS_FILE="1-universe/EVENTS/202603-Hormuz/STATUS.md"
LOG="${PROJECT_DIR}/data/logs/pipeline.log"

# 取最后一行 pipeline log
LAST=$(tail -1 "$LOG" 2>/dev/null)
if [ -z "$LAST" ]; then
    echo "No pipeline log found"
    exit 0
fi

# 解析 JSON
TS=$(echo "$LAST" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['ts'][5:])")  # MM-DD HH:MM
H1=$(echo "$LAST" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"ach\"][\"h1\"]:.0%}')" 2>/dev/null || echo "?")
H2=$(echo "$LAST" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"ach\"][\"h2\"]:.0%}')" 2>/dev/null || echo "?")
T_EXP=$(echo "$LAST" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"t_expected\"]:.0f}')" 2>/dev/null || echo "?")
P50=$(echo "$LAST" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"t_p50\"]:.0f}')" 2>/dev/null || echo "?")
PA=$(echo "$LAST" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"paths\"][\"a\"]:.0%}')" 2>/dev/null || echo "?")
PB=$(echo "$LAST" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"paths\"][\"b\"]:.0%}')" 2>/dev/null || echo "?")
PC=$(echo "$LAST" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"paths\"][\"c\"]:.0%}')" 2>/dev/null || echo "?")
GAP=$(echo "$LAST" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"gap\"]:.0f}')" 2>/dev/null || echo "?")
NEW=$(echo "$LAST" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('articles_new','?'))" 2>/dev/null || echo "?")

# 冲突天数
CONFLICT_START="2026-03-01"
DAY=$(python3 -c "from datetime import datetime; print((datetime.now()-datetime.strptime('$CONFLICT_START','%Y-%m-%d')).days)")

# Brent
BRENT=$(echo "$LAST" | python3 -c "
import json,sys
d=json.load(sys.stdin)
# pipeline log 没存 brent，从 yfinance 快速拉
try:
    import yfinance as yf
    print(f'\${yf.Ticker(\"BZ=F\").info.get(\"regularMarketPrice\",0):.0f}')
except: print('?')
" 2>/dev/null || echo "?")

ROW="| ${TS} | ${DAY} | ${BRENT} | ${H1} | ${H2} | ${T_EXP} | ${P50} | ${PA} | ${PB} | ${PC} | ${GAP} | ${NEW} |"

# Clone octopus (shallow), append row, push
rm -rf "$OCTOPUS_DIR"
git clone --depth 1 https://github.com/CoinSummer/octopus.git "$OCTOPUS_DIR" 2>/dev/null

echo "$ROW" >> "${OCTOPUS_DIR}/${STATUS_FILE}"

cd "$OCTOPUS_DIR"
git add "$STATUS_FILE"
git commit -m "auto: hormuz status ${TS}" --no-gpg-sign 2>/dev/null || exit 0
git push origin main 2>/dev/null

rm -rf "$OCTOPUS_DIR"
echo "Octopus STATUS.md updated: ${TS}"

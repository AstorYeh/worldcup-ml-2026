#!/bin/sh
# 每天定時觸發更新（預設 UTC 02:00 = 台灣 10:00）
# 用 sleep-until-target 取代 cron daemon，最簡單可攜
set -e

TARGET_HOUR="${TARGET_HOUR:-2}"   # 觸發時刻（UTC 小時）
REPO_DIR="${REPO_DIR:-/repo}"

echo "[scheduler] 啟動，每天 UTC ${TARGET_HOUR}:00 更新賠率（repo=${REPO_DIR}）"

# 啟動時先跑一次（確保部署當下就有最新賠率）
/app/update-odds.sh || echo "[scheduler] 首次更新失敗，明日重試"

while true; do
  now=$(date -u +%s)
  next=$(date -u -d "today ${TARGET_HOUR}:00" +%s)
  [ "$next" -le "$now" ] && next=$(date -u -d "tomorrow ${TARGET_HOUR}:00" +%s)
  wait_s=$((next - now))
  echo "[scheduler] 下次更新還有 $((wait_s / 3600)) 小時 $(((wait_s % 3600) / 60)) 分"
  sleep "$wait_s"
  /app/update-odds.sh || echo "[scheduler] 更新失敗，明日重試"
done

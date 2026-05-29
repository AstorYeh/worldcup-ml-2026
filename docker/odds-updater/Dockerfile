# 2026 世界盃冠軍賠率每日自動更新器
# 每天凌晨抓 The Odds API → 覆寫 market_odds.py → git commit & push
FROM python:3.12-slim

# git 用於 commit/push；其餘只用 Python stdlib
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY update-odds.sh /app/update-odds.sh
COPY scheduler.sh   /app/scheduler.sh
RUN chmod +x /app/update-odds.sh /app/scheduler.sh

# 由 scheduler 每天定時觸發 update-odds
ENTRYPOINT ["/app/scheduler.sh"]

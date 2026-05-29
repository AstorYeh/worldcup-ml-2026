# 🎰 賠率每日自動更新器（Docker）

每天凌晨自動抓 The Odds API 的世界盃冠軍盤口賠率，
覆寫 `market_odds.py`，並 `git push` 回 repo →
Streamlit Cloud 自動重新部署 → 奪冠頁的「市場校準層」就會用最新賠率。

> 這是 GitHub Actions `update-odds.yml` 的替代方案 —— 因為某些
> GitHub token 沒有 `workflow` 權限無法建立 Actions，但 Docker cron
> 推的是一般檔案（`market_odds.py`），不受此限。

## 🚀 快速啟動

```bash
cd docker/odds-updater
cp .env.example .env          # 填入 ODDS_API_KEY 與 GITHUB_TOKEN
docker compose up -d --build  # 背景啟動
docker compose logs -f        # 看更新狀態
```

## ⚙️ 運作方式

| 元件 | 功能 |
|------|------|
| `scheduler.sh` | 每天 UTC `TARGET_HOUR`:00 觸發一次（啟動時先跑一次）|
| `update-odds.sh` | git pull → `python fetch_market_odds.py` → 有變動才 commit & push |
| `fetch_market_odds.py` | 抓 The Odds API、隊名映射、多家盤口聚合、覆寫 `market_odds.py` |

## 🔑 必要環境變數（填在 `.env`）

| 變數 | 說明 |
|------|------|
| `ODDS_API_KEY` | The Odds API 金鑰（免費月 500 次）|
| `GITHUB_TOKEN` | GitHub PAT，需 repo push 權限 |
| `GITHUB_REPO` | `owner/name`（預設 AstorYeh/worldcup-ml-2026）|
| `TARGET_HOUR` | 每日觸發 UTC 時刻（預設 2 = 台灣 10:00）|

## 🧪 手動測試一次

```bash
docker compose run --rm odds-updater /app/update-odds.sh
```

## 💡 沒有 API key 也能跑

`fetch_market_odds.py` 在無 `ODDS_API_KEY` 時會優雅跳過、不改檔，
app 沿用現有 `market_odds.py` 快照。要手動更新賠率時，直接編輯
`market_odds.py` 的 `MARKET_ODDS` 與 `AS_OF` 即可。

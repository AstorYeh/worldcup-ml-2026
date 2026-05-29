#!/bin/sh
# 抓最新賠率 → 覆寫 market_odds.py → 若有變動則 commit & push
# git 認證：用 GITHUB_TOKEN 組成的 URL 直接 push，不修改掛載 repo 的 .git/config
set -e

REPO_DIR="${REPO_DIR:-/repo}"
BRANCH="${GIT_BRANCH:-main}"
GH_REPO="${GITHUB_REPO:-AstorYeh/worldcup-ml-2026}"
GH_USER="${GIT_USER_NAME:-odds-bot}"
GH_EMAIL="${GIT_USER_EMAIL:-odds-bot@local}"
cd "$REPO_DIR"

echo "[update] $(date -u '+%Y-%m-%d %H:%M:%S UTC') 開始"

# 認證 URL（有 token 才能 push）
if [ -n "$GITHUB_TOKEN" ]; then
  PUSH_URL="https://${GITHUB_TOKEN}@github.com/${GH_REPO}.git"
else
  PUSH_URL="origin"
  echo "[update] ⚠️ 未設定 GITHUB_TOKEN，將無法 push（仍會更新本地檔案）"
fi

# 安全目錄（掛載 repo 擁有者可能與容器 user 不同）
git config --global --add safe.directory "$REPO_DIR" 2>/dev/null || true

# 1) 同步遠端
git pull --rebase --autostash origin "$BRANCH" 2>/dev/null || echo "[update] git pull 略過"

# 2) 抓賠率（無 ODDS_API_KEY 時 fetch 會優雅跳過、不改檔）
python fetch_market_odds.py

# 3) 有變動才 commit & push（用 -c 注入身分，不動掛載 repo 設定）
if git diff --quiet market_odds.py; then
  echo "[update] 賠率無變動，略過 commit"
else
  git -c user.name="$GH_USER" -c user.email="$GH_EMAIL" \
      commit -am "chore(odds): Docker 每日自動更新世界盃冠軍賠率快照"
  git push "$PUSH_URL" "HEAD:$BRANCH"
  echo "[update] ✅ 已推送更新"
fi

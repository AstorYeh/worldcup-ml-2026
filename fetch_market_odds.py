"""
fetch_market_odds.py — 每日自動抓取世界盃冠軍盤口賠率
============================================================================
從 The Odds API 抓 2026 世界盃冠軍盤口，覆寫 market_odds.py 的
MARKET_ODDS 與 AS_OF，達成「每天凌晨自動浮動更新」。

由 .github/workflows/update-odds.yml 每日 cron 觸發。
需要環境變數 ODDS_API_KEY（GitHub repo secret）。
免費方案：https://the-odds-api.com（每月 500 次，每天 1 次綽綽有餘）

本地手動測試：
    ODDS_API_KEY=xxxx python fetch_market_odds.py

無 key 或抓取失敗時：直接退出、不動 market_odds.py（沿用現有快照）。
"""
import os
import re
import sys
import json
import datetime
import urllib.request

API_KEY = os.environ.get("ODDS_API_KEY", "").strip()
SPORT = "soccer_fifa_world_cup_winner"
URL = (f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
       f"?apiKey={API_KEY}&regions=us&markets=outrights&oddsFormat=decimal")

# The Odds API 隊名 → 本專案 constants.py 隊名
NAME_MAP = {
    "United States": "USA", "South Korea": "South Korea",
    "Korea Republic": "South Korea", "IR Iran": "Iran",
    "Cote d'Ivoire": "Ivory Coast", "Ivory Coast": "Ivory Coast",
    "Turkey": "Turkiye", "Türkiye": "Turkiye",
    "Czech Republic": "Czechia", "Czechia": "Czechia",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "DR Congo": "DR Congo", "Congo DR": "DR Congo",
    "Cape Verde": "Cape Verde", "Cabo Verde": "Cape Verde",
    "Curacao": "Curacao", "Curaçao": "Curacao",
}


def main() -> int:
    if not API_KEY:
        print("⚠️ 未設定 ODDS_API_KEY，跳過更新（沿用現有快照）")
        return 0
    try:
        with urllib.request.urlopen(URL, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"⚠️ 抓取失敗（沿用現有快照）：{e}")
        return 0

    # 聚合所有 bookmaker 的 outrights，同隊取最佳（最低）賠率的平均
    from collections import defaultdict
    acc = defaultdict(list)
    for event in data:
        for bk in event.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "outrights":
                    continue
                for oc in mk.get("outcomes", []):
                    team = NAME_MAP.get(oc["name"], oc["name"])
                    price = oc.get("price")
                    if price and price > 1:
                        acc[team].append(float(price))
    if not acc:
        print("⚠️ API 回傳無 outrights 資料（沿用現有快照）")
        return 0
    odds = {t: round(sum(v) / len(v), 1) for t, v in acc.items()}

    # 讀現有 market_odds.py，保留未被 API 覆蓋的隊伍（長盤段）
    import market_odds as M
    merged = dict(M.MARKET_ODDS)
    merged.update(odds)

    today = datetime.date.today().isoformat()
    _write_market_odds(merged, today)
    print(f"✅ 已更新 {len(odds)} 隊賠率，AS_OF={today}")
    return 0


def _write_market_odds(odds: dict, as_of: str) -> None:
    """以新賠率重寫 market_odds.py 的 AS_OF 與 MARKET_ODDS（保留其餘程式碼）。"""
    src = open("market_odds.py", encoding="utf-8").read()
    # 更新 AS_OF
    src = re.sub(r'AS_OF\s*=\s*"[^"]*"', f'AS_OF = "{as_of}"', src, count=1)
    # 重建 MARKET_ODDS 區塊（依賠率升序）
    lines = ["MARKET_ODDS = {"]
    for t, o in sorted(odds.items(), key=lambda x: x[1]):
        lines.append(f"    {t!r}: {o},")
    lines.append("}")
    new_block = "\n".join(lines)
    src = re.sub(r"MARKET_ODDS\s*=\s*\{.*?\n\}", new_block, src,
                 count=1, flags=re.DOTALL)
    open("market_odds.py", "w", encoding="utf-8").write(src)


if __name__ == "__main__":
    sys.exit(main())

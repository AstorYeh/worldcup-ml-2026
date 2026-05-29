"""
market_odds.py — 2026 世界盃冠軍盤口快照（市場情緒量化來源）
============================================================================
把「博彩市場共識」轉成可計算的數字：十進位賠率 → 去抽水隱含機率。

這是 app.py 奪冠預測頁「市場校準層」的資料來源。

🔄 自動更新：docker/odds-updater/ 容器每天凌晨用 The Odds API
   重抓賠率並覆寫本檔 MARKET_ODDS / AS_OF，再 git push。
   （未設定 API key 時，沿用此手動快照 — app 仍正常運作）

⚠️ 賠率為十進位（decimal）：美式 +N → N/100 + 1（例：+430 → 5.30）
"""

# 快照時點（自動更新會改寫此日期）
AS_OF = "2026-05-26"
# 開賽日（用於融合權重的時間衰減）
TOURNAMENT_START = "2026-06-11"

# 各家運彩世界盃冠軍盤口共識（十進位賠率，數字越小 = 越被看好）
# Top 段為 2026/5 真實市場共識；長盤段依 FIFA 實力分層估計
MARKET_ODDS = {
    # ── 熱門段（真實市場賠率）──
    'Spain': 5.3, 'France': 6.0, 'England': 7.5, 'Brazil': 9.0, 'Argentina': 9.5,
    'Germany': 11.0, 'Portugal': 13.0, 'Netherlands': 13.0, 'Belgium': 17.0,
    # ── 中段 ──
    'Uruguay': 26.0, 'Croatia': 29.0, 'Colombia': 34.0, 'Morocco': 41.0,
    'Japan': 51.0, 'Switzerland': 51.0, 'Senegal': 67.0,
    'Mexico': 67.0, 'USA': 67.0, 'South Korea': 81.0, 'Ecuador': 81.0,
    # ── 長盤段（依 FIFA 實力分層估計）──
    'Sweden': 101.0, 'Austria': 101.0, 'Turkiye': 101.0, 'Norway': 126.0,
    'Australia': 151.0, 'Ivory Coast': 151.0, 'Egypt': 151.0, 'Iran': 151.0,
    'Algeria': 201.0, 'Paraguay': 201.0, 'Scotland': 201.0, 'Canada': 201.0,
    'Tunisia': 251.0, 'Qatar': 251.0, 'Saudi Arabia': 301.0, 'Uzbekistan': 301.0,
    'Czechia': 301.0, 'DR Congo': 351.0, 'Panama': 401.0, 'Iraq': 401.0,
    'Jordan': 501.0, 'South Africa': 501.0, 'Cape Verde': 751.0,
    'Bosnia and Herzegovina': 751.0, 'Ghana': 501.0, 'New Zealand': 751.0,
    'Haiti': 1001.0, 'Curacao': 1001.0,
}


def market_implied_probs() -> dict:
    """十進位賠率 → 去抽水（de-vig）後的市場奪冠機率。

    步驟：
        1. 隱含機率 = 1 / 賠率
        2. overround = Σ 所有隊隱含機率（含莊家抽水，通常 > 1）
        3. 真實機率 = 隱含機率 / overround   ← 移除抽水，正規化到總和=1
    回傳 {隊名: 機率(0~1)}
    """
    raw = {t: 1.0 / o for t, o in MARKET_ODDS.items() if o and o > 0}
    overround = sum(raw.values())
    if overround <= 0:
        return {}
    return {t: r / overround for t, r in raw.items()}


def market_overround() -> float:
    """回傳莊家抽水比例（overround − 1），供透明顯示。"""
    raw = sum(1.0 / o for o in MARKET_ODDS.values() if o and o > 0)
    return raw - 1.0

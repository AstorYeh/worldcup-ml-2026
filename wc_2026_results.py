# -*- coding: utf-8 -*-
"""抓取 2026 世界盃「已完成」賽果（ESPN 公開 API），轉成與 match_df 相容的 DataFrame，
供模型微調（把實戰表現灌回各隊近期攻防/狀態）。失敗時回傳空 DataFrame，不影響主流程。"""
import json
import urllib.request

import pandas as pd

# ESPN 隊名 → 專案內部鍵值（僅列與內部不同者）
_ESPN_NAME_MAP = {
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Congo DR': 'DR Congo',
    'Curaçao': 'Curacao',
    'Türkiye': 'Turkiye',
    'United States': 'USA',
}

_COLS = ['date', 'home_team', 'away_team', 'home_score', 'away_score',
         'tournament', 'neutral', 'year']


def fetch_2026_wc_results() -> pd.DataFrame:
    """回傳已完成 2026 WC 小組賽結果（match_df 相容欄位）。"""
    url = ('https://site.api.espn.com/apis/site/v2/sports/soccer/'
           'fifa.world/scoreboard?dates=20260611-20260720')
    rows = []
    try:
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = json.load(op.open(req, timeout=15))
    except Exception:
        return pd.DataFrame(columns=_COLS)

    for e in data.get('events', []):
        comp = (e.get('competitions') or [{}])[0]
        cs = comp.get('competitors', [])
        if len(cs) != 2:
            continue
        state = ((e.get('status') or {}).get('type') or {}).get('state')
        if state != 'post':          # 只取已完賽
            continue
        home = away = hs = as_ = None
        ok = True
        for c in cs:
            raw = (c.get('team') or {}).get('displayName', '')
            nm = _ESPN_NAME_MAP.get(raw, raw)
            try:
                sc = int(c.get('score'))
            except (TypeError, ValueError):
                ok = False
                break
            if c.get('homeAway') == 'home':
                home, hs = nm, sc
            else:
                away, as_ = nm, sc
        if not ok or home is None or away is None:
            continue
        rows.append({
            'date': pd.to_datetime((e.get('date') or '')[:10], errors='coerce'),
            'home_team': home, 'away_team': away,
            'home_score': hs, 'away_score': as_,
            'tournament': 'FIFA World Cup', 'neutral': True, 'year': 2026,
        })
    return pd.DataFrame(rows, columns=_COLS)


if __name__ == '__main__':
    df = fetch_2026_wc_results()
    print(f"已完成 {len(df)} 場：")
    for _, r in df.iterrows():
        print(f"  {r['home_team']} {r['home_score']}-{r['away_score']} {r['away_team']}")

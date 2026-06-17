"""
🏆 Pretrain Script — 2026 World Cup ML
========================================
把所有 ML 任務 offline 跑一遍，產出 pkl 檔供 app.py 直接讀取，
徹底消除 Streamlit Cloud 冷啟動 30 秒等待。

執行：python pretrain.py（約 1-2 分鐘）

產出 models/ 內 8 個 pkl：
    clf.pkl              XGBoost 三向分類器
    poisson1.pkl         Team1 進球 Poisson
    poisson2.pkl         Team2 進球 Poisson
    feat_cols.pkl        特徵欄位列表
    val_accs.pkl         Walk-Forward 4 屆 WC 驗證準確率
    mc_results.pkl       Monte Carlo 10,000 次奪冠機率
    team_clusters.pkl    K-Means + PCA 球隊風格分群
    eval_metrics.pkl     混淆矩陣 / ROC / 校準 / 特徵重要性

副產品（仅本機除錯用，已在 .gitignore）：
    figures/14-18_*.png  matplotlib 靜態圖（app.py 不使用，全用 Plotly 重畫）
    data_cache/*.pkl     歷史資料快取
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import poisson as scipy_poisson

warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(REPO_ROOT, 'models')
DATA_CACHE = os.path.join(REPO_ROOT, 'data_cache')
FIG_DIR = os.path.join(REPO_ROOT, 'figures')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_CACHE, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# 球隊資料：全部從 constants.py 取（與 app.py 共用單一來源）
# ============================================================
from constants import (
    TEAM_INFO, WC_2026_GROUPS, CONFED_MAP, CONFED_BONUS,
    TEAM_PTS, TEAM_RANK,
)
from squad_data import SQUAD_DATA

ALL_TEAMS = sorted({t for teams in WC_2026_GROUPS.values() for t in teams})

# ── 主將陣容平均 OVR（Monte Carlo 注入用，與 app.py predict_match 同邏輯）──
_SQUAD_OVR_BASELINE = 79.0

def squad_ovr(team):
    """回傳球隊主將平均 OVR；無資料則回基準值 79。"""
    players = SQUAD_DATA.get(team, [])
    if not players:
        return _SQUAD_OVR_BASELINE
    return float(np.mean([p['ovr'] for p in players]))


# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    cache_match = os.path.join(DATA_CACHE, 'match_df.pkl')
    cache_fifa = os.path.join(DATA_CACHE, 'fifa_df.pkl')

    if os.path.exists(cache_match) and os.path.exists(cache_fifa):
        print("[Cache] 從本地 cache 載入資料")
        with open(cache_match, 'rb') as f:
            match_df = pickle.load(f)
        with open(cache_fifa, 'rb') as f:
            fifa_df = pickle.load(f)
    else:
        # 先試本地 fallback（沙盒環境用），再退到 URL
        local_match = '/tmp/intl_data/results.csv'
        local_fifa  = '/tmp/fifa_data/ranking_fifa_historical.csv'
        if os.path.exists(local_match):
            print("[Local] 從 /tmp/intl_data 載入 results.csv")
            match_df = pd.read_csv(local_match)
        else:
            print("[Download] 從 GitHub 下載 results.csv")
            match_df = pd.read_csv(
                "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
            )
        match_df['date'] = pd.to_datetime(match_df['date'])
        match_df['year'] = match_df['date'].dt.year
        match_df = match_df.dropna(subset=['home_score', 'away_score'])
        match_df['home_score'] = match_df['home_score'].astype(int)
        match_df['away_score'] = match_df['away_score'].astype(int)
        with open(cache_match, 'wb') as f:
            pickle.dump(match_df, f)

        if os.path.exists(local_fifa):
            print("[Local] 從 /tmp/fifa_data 載入 ranking_fifa_historical.csv")
            fifa_df = pd.read_csv(local_fifa)
        else:
            print("[Download] 從 GitHub 下載 fifa-ranking")
            fifa_df = pd.read_csv(
                "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/ranking_fifa_historical.csv"
            )
        fifa_df['date'] = pd.to_datetime(fifa_df['date'])
        with open(cache_fifa, 'wb') as f:
            pickle.dump(fifa_df, f)

    # 注入 2026 世界盃已完成賽果（供 2026 強度微調；訓練集排除 WC 故不受影響）
    try:
        from wc_2026_results import fetch_2026_wc_results
        wc26 = fetch_2026_wc_results()
        if len(wc26) > 0:
            wc26 = wc26.reindex(columns=match_df.columns)
            match_df = pd.concat([match_df, wc26], ignore_index=True)
            print(f"[2026] 已注入 {len(wc26)} 場世界盃實戰賽果（微調用）")
    except Exception as _e:
        print(f"[2026] 注入賽果失敗（略過，用純歷史）：{_e}")

    print(f"[Data] match_df = {len(match_df):,} 場；fifa_df = {len(fifa_df):,} 筆排名")
    return match_df, fifa_df


# ============================================================
# FEATURE ENGINEERING（與 app.py 一致）
# ============================================================
def compute_team_strength(df, team, year, years_back=8, include_current=False):
    start_year = year - years_back
    upper = (df['year'] <= year) if include_current else (df['year'] < year)
    tm = df[((df['home_team'] == team) | (df['away_team'] == team)) &
            (df['year'] >= start_year) & upper].copy()
    if len(tm) == 0:
        return {'win_rate': 0.35, 'draw_rate': 0.25, 'avg_goals': 1.2, 'avg_conceded': 1.3, 'matches': 0}
    home_mask = tm['home_team'] == team
    gf = np.where(home_mask, tm['home_score'], tm['away_score']).astype(float)
    ga = np.where(home_mask, tm['away_score'], tm['home_score']).astype(float)
    weight = np.exp(-0.1 * (year - tm['year'].values))
    tw = weight.sum()
    win_rate = (weight * (gf > ga)).sum() / tw
    draw_rate = (weight * (gf == ga)).sum() / tw
    return {
        'win_rate': float(win_rate),
        'draw_rate': float(draw_rate),
        'avg_goals': float((gf * weight).sum() / tw),
        'avg_conceded': float((ga * weight).sum() / tw),
        'matches': int(len(tm)),
    }


def compute_recent_form(df, team, year, include_current=False):
    upper = (df['year'] <= year) if include_current else (df['year'] < year)
    recent = df[((df['home_team'] == team) | (df['away_team'] == team)) &
                (df['year'] >= year - 2) & upper]
    if len(recent) == 0:
        return 0.35
    home_mask = recent['home_team'] == team
    gf = np.where(home_mask, recent['home_score'], recent['away_score'])
    ga = np.where(home_mask, recent['away_score'], recent['home_score'])
    return float((gf > ga).mean())


def compute_knockout_exp(df, team):
    wc = df[(df['tournament'].str.contains('World Cup', na=False)) &
            (~df['tournament'].str.contains('qualif', case=False, na=False)) &
            ((df['home_team'] == team) | (df['away_team'] == team))]
    return int(len(wc))


def get_historical_ranking(fifa_df, team, date):
    row = fifa_df[(fifa_df['team'] == team) & (fifa_df['date'] <= date)].sort_values('date').tail(1)
    if len(row) == 0:
        return 1500.0
    return float(row.iloc[0]['total_points'])


def create_features_v2(team1, team2, year, match_df, fifa_df,
                       strength_cache=None, form_cache=None, ko_cache=None,
                       include_current=False):
    """支援快取版本，加速大量訓練。include_current=True 時把當年(2026)實戰也算進強度。"""
    if strength_cache is None or (team1, year) not in strength_cache:
        s1 = compute_team_strength(match_df, team1, year, include_current=include_current)
        if strength_cache is not None:
            strength_cache[(team1, year)] = s1
    else:
        s1 = strength_cache[(team1, year)]

    if strength_cache is None or (team2, year) not in strength_cache:
        s2 = compute_team_strength(match_df, team2, year, include_current=include_current)
        if strength_cache is not None:
            strength_cache[(team2, year)] = s2
    else:
        s2 = strength_cache[(team2, year)]

    pts1 = get_historical_ranking(fifa_df, team1, f"{year}-05-01")
    pts2 = get_historical_ranking(fifa_df, team2, f"{year}-05-01")

    r1 = TEAM_RANK.get(team1, 99)
    r2 = TEAM_RANK.get(team2, 99)

    confed1 = CONFED_MAP.get(team1, 'CAF')
    confed2 = CONFED_MAP.get(team2, 'CAF')

    if form_cache is None or (team1, year) not in form_cache:
        f1 = compute_recent_form(match_df, team1, year, include_current=include_current)
        if form_cache is not None: form_cache[(team1, year)] = f1
    else:
        f1 = form_cache[(team1, year)]
    if form_cache is None or (team2, year) not in form_cache:
        f2 = compute_recent_form(match_df, team2, year, include_current=include_current)
        if form_cache is not None: form_cache[(team2, year)] = f2
    else:
        f2 = form_cache[(team2, year)]

    if ko_cache is None or team1 not in ko_cache:
        k1 = compute_knockout_exp(match_df, team1)
        if ko_cache is not None: ko_cache[team1] = k1
    else:
        k1 = ko_cache[team1]
    if ko_cache is None or team2 not in ko_cache:
        k2 = compute_knockout_exp(match_df, team2)
        if ko_cache is not None: ko_cache[team2] = k2
    else:
        k2 = ko_cache[team2]

    return {
        'pts_diff': pts1 - pts2,
        'rank_diff': r1 - r2,
        'win_rate_diff': s1['win_rate'] - s2['win_rate'],
        'avg_goals_diff': s1['avg_goals'] - s2['avg_goals'],
        'defense_diff': s1['avg_conceded'] - s2['avg_conceded'],
        'form_diff': (s1['win_rate']*s1['avg_goals']) - (s2['win_rate']*s2['avg_goals']),
        'experience': s1['matches'] + s2['matches'],
        'goals_product': s1['avg_goals'] * s2['avg_goals'],
        'confed_bonus': CONFED_BONUS.get(confed1, 0) - CONFED_BONUS.get(confed2, 0),
        'recent_form': f1 - f2,
        'knockout_exp': k1 - k2,
    }


def make_intl_dataset(match_df, fifa_df, year, years_back=36):
    """組訓練樣本：以該年之前歷年國際賽（排除 WC）為主"""
    start_year = year - years_back
    intl = match_df[(match_df['year'] >= start_year) & (match_df['year'] < year) &
                    (~match_df['tournament'].str.contains('World Cup', na=False))]
    intl = intl[(intl['home_team'].isin(ALL_TEAMS)) & (intl['away_team'].isin(ALL_TEAMS))]
    intl = intl[intl['year'] >= 1990]

    strength_cache = {}
    form_cache = {}
    ko_cache = {}

    samples = []
    rows = intl.to_dict('records')
    for i, row in enumerate(rows):
        if i % 1000 == 0:
            print(f"  ... {i}/{len(rows)}")
        t1, t2, yr = row['home_team'], row['away_team'], row['year']
        try:
            feat = create_features_v2(t1, t2, yr, match_df, fifa_df,
                                      strength_cache, form_cache, ko_cache)
            gf, ga = row['home_score'], row['away_score']
            samples.append({
                'feat': feat,
                'label': 0 if gf < ga else (1 if gf == ga else 2),
                'gf': int(gf), 'ga': int(ga),
                'team1': t1, 'team2': t2, 'year': yr,
            })
        except Exception:
            continue
    return samples


def make_wc_dataset(match_df, fifa_df, year):
    """指定年世界盃小組賽作驗證集（用當年實際 WC 隊伍，不限 2026 分組）"""
    samples = []
    strength_cache, form_cache, ko_cache = {}, {}, {}
    wc = match_df[
        (match_df['year'] == year) &
        (match_df['tournament'].str.contains('World Cup', na=False)) &
        (~match_df['tournament'].str.contains('qualif', case=False, na=False))
    ]
    for _, row in wc.iterrows():
        t1, t2 = row['home_team'], row['away_team']
        try:
            feat = create_features_v2(t1, t2, year, match_df, fifa_df,
                                      strength_cache, form_cache, ko_cache)
            gf, ga = int(row['home_score']), int(row['away_score'])
            samples.append({
                'feat': feat,
                'label': 0 if gf < ga else (1 if gf == ga else 2),
                'gf': gf, 'ga': ga,
                'team1': t1, 'team2': t2, 'year': year,
            })
        except Exception:
            continue
    return samples


# ============================================================
# 1) 訓練分類 + 迴歸模型
# ============================================================
def train_models(match_df, fifa_df):
    print("\n[1/4] 訓練 XGBoost + Poisson 模型（walk-forward 驗證）")
    train_final = make_intl_dataset(match_df, fifa_df, 2026, years_back=36)
    print(f"  訓練集大小：{len(train_final)}")

    X_tr = pd.DataFrame([s['feat'] for s in train_final])
    y_tr = np.array([s['label'] for s in train_final])
    y_g1 = np.array([s['gf'] for s in train_final])
    y_g2 = np.array([s['ga'] for s in train_final])
    feat_cols = list(X_tr.columns)

    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='mlogloss',
    )
    clf.fit(X_tr, y_tr)
    print(f"  ✅ XGBoost 訓練完成（train acc={accuracy_score(y_tr, clf.predict(X_tr)):.3f}）")

    poisson1 = PoissonRegressor(alpha=0.1, max_iter=500).fit(X_tr, y_g1)
    poisson2 = PoissonRegressor(alpha=0.1, max_iter=500).fit(X_tr, y_g2)
    print(f"  ✅ Poisson Team1/Team2 訓練完成")

    # Walk-forward 驗證
    val_accs = {}
    for yr in [2010, 2014, 2018, 2022]:
        val = make_wc_dataset(match_df, fifa_df, yr)
        if len(val) >= 5:
            Xv = pd.DataFrame([s['feat'] for s in val])[feat_cols]
            yv = np.array([s['label'] for s in val])
            val_accs[yr] = float(accuracy_score(yv, clf.predict(Xv)))
            print(f"  WC {yr} val acc = {val_accs[yr]:.3f}")
        else:
            val_accs[yr] = None

    # 存
    with open(os.path.join(MODEL_DIR, 'clf.pkl'), 'wb') as f: pickle.dump(clf, f)
    with open(os.path.join(MODEL_DIR, 'poisson1.pkl'), 'wb') as f: pickle.dump(poisson1, f)
    with open(os.path.join(MODEL_DIR, 'poisson2.pkl'), 'wb') as f: pickle.dump(poisson2, f)
    with open(os.path.join(MODEL_DIR, 'feat_cols.pkl'), 'wb') as f: pickle.dump(feat_cols, f)
    with open(os.path.join(MODEL_DIR, 'val_accs.pkl'), 'wb') as f: pickle.dump(val_accs, f)
    print(f"  💾 models/ 存好了")

    return clf, poisson1, poisson2, feat_cols, val_accs


# ============================================================
# 2) Monte Carlo 奪冠機率
# ============================================================
def run_monte_carlo(match_df, fifa_df, clf, poisson1, poisson2, feat_cols, n_sims=10000):
    print(f"\n[2/4] Monte Carlo {n_sims} 次模擬")

    # ── 主將陣容修正（v2 新增）──
    # 原本 MC 完全靠 FIFA 積分 + XGBoost，零陣容資訊 → 南美被高估、法國等被低估。
    # 對每組對戰的期望進球 λ 乘上 (ovr1/ovr2)^0.35，把「市場最看重的陣容」納入。
    #
    # ⚠️ 設計說明：為何這裡「只」注入 squad、而非 predict_match 的完整四因素？
    #   MC 的 λ 來自 Poisson 迴歸（poisson1/2），其特徵已含 pts_diff / recent_form /
    #   knockout_exp → FIFA 積分、近期狀態、淘汰賽經驗「已被模型學進去」。
    #   若在此再乘 rank/form/exp factor 會「重複計算」。唯獨 squad OVR 不在特徵裡，
    #   故只補這一項。predict_match 的基底是 Dixon-Coles 純攻防公式（無這些特徵），
    #   才需要完整四因素 composite。兩條路徑因「基底不同」而修正項不同，並非 bug。
    SQUAD_POW = 0.35
    def _squad_adjust(t1, t2, lam1, lam2):
        sf = (squad_ovr(t1) / max(squad_ovr(t2), 1.0)) ** SQUAD_POW
        return max(0.1, lam1 * sf), max(0.1, lam2 / sf)

    # 預先算 2026 所有可能對戰的勝率與進球期望
    strength_cache, form_cache, ko_cache = {}, {}, {}
    pairs = {}
    for g, teams in WC_2026_GROUPS.items():
        for i, t1 in enumerate(teams):
            for t2 in teams[i+1:]:
                feat = create_features_v2(t1, t2, 2026, match_df, fifa_df,
                                          strength_cache, form_cache, ko_cache,
                                          include_current=True)
                X = pd.DataFrame([feat])[feat_cols]
                prob = clf.predict_proba(X)[0]  # [L, D, W from team1]
                lam1 = max(0.1, float(poisson1.predict(X)[0]))
                lam2 = max(0.1, float(poisson2.predict(X)[0]))
                lam1, lam2 = _squad_adjust(t1, t2, lam1, lam2)
                pairs[(t1, t2)] = (prob, lam1, lam2)
                pairs[(t2, t1)] = (prob[[2,1,0]], lam2, lam1)

    # 跨組對戰也需要：淘汰賽
    for ta in ALL_TEAMS:
        for tb in ALL_TEAMS:
            if ta == tb or (ta, tb) in pairs:
                continue
            try:
                feat = create_features_v2(ta, tb, 2026, match_df, fifa_df,
                                          strength_cache, form_cache, ko_cache,
                                          include_current=True)
                X = pd.DataFrame([feat])[feat_cols]
                prob = clf.predict_proba(X)[0]
                lam1 = max(0.1, float(poisson1.predict(X)[0]))
                lam2 = max(0.1, float(poisson2.predict(X)[0]))
                lam1, lam2 = _squad_adjust(ta, tb, lam1, lam2)
                pairs[(ta, tb)] = (prob, lam1, lam2)
            except Exception:
                continue

    rng = np.random.default_rng(42)

    def sim_match(ta, tb):
        """回傳 (winner, score_a, score_b)；用 Poisson 抽進球，平手用勝率裁決"""
        prob, l1, l2 = pairs[(ta, tb)]
        ga = rng.poisson(l1)
        gb = rng.poisson(l2)
        if ga > gb: return ta, ga, gb
        if gb > ga: return tb, ga, gb
        # 平手：用分類器勝率裁決（PK 邏輯，正規化避免浮點誤差）
        pw, pl = float(prob[2]), float(prob[0])
        if pw + pl < 1e-9:
            pw, pl = 0.5, 0.5
        s = pw + pl
        winner = rng.choice([ta, tb], p=[pw/s, pl/s])
        return winner, ga, gb

    # 已完賽結果（從 match_df 的 2026 WC 列取出）→ 小組賽鎖定真實結果，只模擬未賽場次
    _t2g = {t: g for g, ts in WC_2026_GROUPS.items() for t in ts}
    played = {}
    _wc26 = match_df[(match_df['year'] == 2026) &
                     (match_df['tournament'].astype(str).str.contains('World Cup', na=False))]
    for _, _r in _wc26.iterrows():
        _h, _a = _r['home_team'], _r['away_team']
        if _t2g.get(_h) and _t2g.get(_h) == _t2g.get(_a):
            played[frozenset((_h, _a))] = {_h: int(_r['home_score']), _a: int(_r['away_score'])}
    if played:
        print(f"  [MC] 鎖定 {len(played)} 場已完賽結果（條件化模擬），未賽才模擬")

    # 各組名次分布 / 期望積分 / 最佳第3晉級 統計（供「各組排名預測」頁）
    group_rank_count = {g: {t: [0, 0, 0, 0] for t in ts} for g, ts in WC_2026_GROUPS.items()}
    group_pts_sum = {g: {t: 0 for t in ts} for g, ts in WC_2026_GROUPS.items()}
    group_gd_sum = {g: {t: 0 for t in ts} for g, ts in WC_2026_GROUPS.items()}
    third_adv_count = {t: 0 for t in ALL_TEAMS}

    champion_count = {t: 0 for t in ALL_TEAMS}
    final_count    = {t: 0 for t in ALL_TEAMS}
    semi_count     = {t: 0 for t in ALL_TEAMS}
    quarter_count  = {t: 0 for t in ALL_TEAMS}
    r16_count      = {t: 0 for t in ALL_TEAMS}
    advance_count  = {t: 0 for t in ALL_TEAMS}  # 出小組

    for sim in range(n_sims):
        if sim % 1000 == 0:
            print(f"  sim {sim}/{n_sims}")
        # 小組賽（已賽鎖定真實結果、未賽才模擬）
        group_results = {}
        sim_thirds = []
        for g, teams in WC_2026_GROUPS.items():
            pts = {t: 0 for t in teams}
            gd = {t: 0 for t in teams}
            gf = {t: 0 for t in teams}
            for i, t1 in enumerate(teams):
                for t2 in teams[i+1:]:
                    _pl = played.get(frozenset((t1, t2)))
                    if _pl is not None:                       # 已賽 → 鎖定真實比分
                        g1, g2 = _pl[t1], _pl[t2]
                        w = t1 if g1 > g2 else (t2 if g2 > g1 else None)
                    else:
                        w, g1, g2 = sim_match(t1, t2)
                    gf[t1] += g1; gf[t2] += g2
                    gd[t1] += g1-g2; gd[t2] += g2-g1
                    if w == t1: pts[t1] += 3
                    elif w == t2: pts[t2] += 3
                    else: pts[t1] += 1; pts[t2] += 1
            # 排序
            standing = sorted(teams, key=lambda t: (-pts[t], -gd[t], -gf[t], rng.random()))
            group_results[g] = standing
            for r_idx, t in enumerate(standing):
                group_rank_count[g][t][r_idx] += 1
                group_pts_sum[g][t] += pts[t]
                group_gd_sum[g][t] += gd[t]
            for t in standing[:2]:
                advance_count[t] += 1
            t3 = standing[2]
            sim_thirds.append((t3, pts[t3], gd[t3], gf[t3]))

        # 最佳 8 個第 3 名：依模擬小組積分/淨勝/進球（與真實規則一致）
        sim_thirds.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        best_thirds = [x[0] for x in sim_thirds[:8]]
        for t in best_thirds:
            third_adv_count[t] += 1

        bracket = []  # 32 teams
        # 1st 12 + 2nd 12 + best thirds 8
        firsts = [group_results[g][0] for g in 'ABCDEFGHIJKL']
        seconds = [group_results[g][1] for g in 'ABCDEFGHIJKL']
        bracket = firsts + seconds + best_thirds  # 32
        rng.shuffle(bracket)

        # Round of 32
        for t in bracket:
            r16_count[t] += 1
        next_round = []
        for i in range(0, 32, 2):
            w, _, _ = sim_match(bracket[i], bracket[i+1])
            next_round.append(w)
        # Round of 16
        for t in next_round:
            quarter_count[t] += 1
        nr = []
        for i in range(0, 16, 2):
            w, _, _ = sim_match(next_round[i], next_round[i+1])
            nr.append(w)
        # QF
        for t in nr: semi_count[t] += 1
        nr2 = []
        for i in range(0, 8, 2):
            w, _, _ = sim_match(nr[i], nr[i+1])
            nr2.append(w)
        # SF
        for t in nr2: final_count[t] += 1
        nr3 = []
        for i in range(0, 4, 2):
            w, _, _ = sim_match(nr2[i], nr2[i+1])
            nr3.append(w)
        # Final
        champ, _, _ = sim_match(nr3[0], nr3[1])
        champion_count[champ] += 1

    mc_teams = {t: {
            'champ_pct': champion_count[t] / n_sims * 100,
            'final_pct': final_count[t] / n_sims * 100,
            'semi_pct': semi_count[t] / n_sims * 100,
            'quarter_pct': quarter_count[t] / n_sims * 100,
            'r16_pct': r16_count[t] / n_sims * 100,
            'win_pct': advance_count[t] / n_sims * 100,
          } for t in ALL_TEAMS}
    group_standings = {}
    for g, ts in WC_2026_GROUPS.items():
        group_standings[g] = {}
        for t in ts:
            rc = group_rank_count[g][t]
            group_standings[g][t] = {
                'p1': rc[0] / n_sims * 100, 'p2': rc[1] / n_sims * 100,
                'p3': rc[2] / n_sims * 100, 'p4': rc[3] / n_sims * 100,
                'exp_pts': group_pts_sum[g][t] / n_sims,
                'exp_gd': group_gd_sum[g][t] / n_sims,
                'advance_pct': advance_count[t] / n_sims * 100,      # 前2直接晉級
                'third_adv_pct': third_adv_count[t] / n_sims * 100,  # 以最佳第3晉級
            }
    mc = {'n_sims': n_sims, 'results': mc_teams, 'group_standings': group_standings}

    with open(os.path.join(MODEL_DIR, 'mc_results.pkl'), 'wb') as f:
        pickle.dump(mc, f)
    top10 = sorted(mc_teams.items(), key=lambda x: -x[1]['champ_pct'])[:10]
    print("  Top 10 奪冠機率：")
    for t, d in top10:
        print(f"    {t:25s}  champ={d['champ_pct']:5.2f}%  final={d['final_pct']:5.2f}%  win_grp={d['win_pct']:5.1f}%")
    print(f"  💾 models/mc_results.pkl 存好了")
    return mc_teams


# ============================================================
# 3) 球隊風格分群（KMeans + PCA）
# ============================================================
def run_clustering(match_df):
    print("\n[3/4] 球隊風格分群（KMeans, k=3）")
    # 每隊統計：近 8 年平均進攻、防守、勝率、實力、競賽量
    features = []
    for t in ALL_TEAMS:
        s = compute_team_strength(match_df, t, 2026, years_back=8)
        ko = compute_knockout_exp(match_df, t)
        features.append({
            'team': t,
            'win_rate': s['win_rate'],
            'avg_goals': s['avg_goals'],
            'avg_conceded': s['avg_conceded'],
            'goal_diff': s['avg_goals'] - s['avg_conceded'],
            'matches': s['matches'],
            'knockout_exp': ko,
            'fifa_pts': TEAM_PTS.get(t, 1500),
        })
    df = pd.DataFrame(features).set_index('team')

    # 標準化
    feat_cols = ['win_rate', 'avg_goals', 'avg_conceded', 'goal_diff',
                 'matches', 'knockout_exp', 'fifa_pts']
    X = df[feat_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 固定 k=3（攻擊型 / 防守型 / 平衡型），並記錄 silhouette 供展示
    from sklearn.metrics import silhouette_score
    best_k = 3
    sil = {}
    for k in [3, 4, 5]:
        km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        sil[k] = silhouette_score(Xs, km_tmp.fit_predict(Xs))
        print(f"  k={k} silhouette = {sil[k]:.3f}")
    print(f"  固定使用 k={best_k}（攻擊型/防守型/平衡型）")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    df['cluster'] = labels

    # 為各群定名稱：依攻守特徵（avg_goals / avg_conceded）判斷風格
    centers_unscaled = scaler.inverse_transform(km.cluster_centers_)
    centers_df = pd.DataFrame(centers_unscaled, columns=feat_cols)

    # 判斷邏輯（k=3，各群恰好一個）：
    #   攻擊型 — 淨進球（avg_goals - avg_conceded）最高 → 進攻主導
    #   防守型 — 淨進球最低 → 穩守反擊
    #   平衡型 — 中間 → 攻守均衡
    centers_df['net_goal'] = centers_df['avg_goals'] - centers_df['avg_conceded']
    net_rank = centers_df['net_goal'].rank(ascending=False).astype(int)  # 1=最高淨進球
    cluster_names    = {}
    cluster_names_en = {}
    for c in range(best_k):
        r = net_rank[c]
        if r == 1:
            cluster_names[c]    = '⚡ 攻擊型'
            cluster_names_en[c] = 'Attacking'
        elif r == best_k:
            cluster_names[c]    = '🛡️ 防守型'
            cluster_names_en[c] = 'Defensive'
        else:
            cluster_names[c]    = '⚖️ 平衡型'
            cluster_names_en[c] = 'Balanced'

    # PCA 2D 視覺化
    pca = PCA(n_components=2)
    pca_xy = pca.fit_transform(Xs)
    df['pca1'] = pca_xy[:, 0]
    df['pca2'] = pca_xy[:, 1]

    # 存
    out = {
        'df': df,
        'centers': centers_df,
        'cluster_names': cluster_names,
        'feat_cols': feat_cols,
        'silhouette': sil[best_k],
        'k': best_k,
        'kmeans': km,
        'scaler': scaler,
        'pca': pca,
        'cluster_names_en': cluster_names_en,
    }
    with open(os.path.join(MODEL_DIR, 'team_clusters.pkl'), 'wb') as f:
        pickle.dump(out, f)
    print(f"  💾 models/team_clusters.pkl 存好了")

    # 視覺化（雷達圖 + PCA 散點）
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        plt = None

    if plt is not None:
        # 使用繁體中文字型（本機 Windows 微軟正黑體）
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Apple LiGothic',
                                           'Noto Sans CJK TC', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 雷達圖
        radar_feats = ['win_rate', 'avg_goals', 'avg_conceded', 'goal_diff', 'fifa_pts']
        # 將每個 cluster 中心 normalize 到 0~1
        c_data = centers_df[radar_feats].copy()
        c_norm = (c_data - c_data.min()) / (c_data.max() - c_data.min() + 1e-9)
        c_norm['avg_conceded'] = 1 - c_norm['avg_conceded']  # 反向：越小越好 → 越大越好

        angles = np.linspace(0, 2*np.pi, len(radar_feats), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = ['#e94560', '#f7c59f', '#00d4ff', '#7c4dff', '#26de81']
        for c in range(best_k):
            vals = c_norm.iloc[c].tolist()
            vals += vals[:1]
            ax.plot(angles, vals, 'o-', linewidth=2, label=cluster_names[c].replace('🔥','').replace('⚔️','').replace('🛡️','').replace('🌱','').strip(), color=colors[c % len(colors)])
            ax.fill(angles, vals, alpha=0.15, color=colors[c % len(colors)])
        ax.set_thetagrids(np.degrees(angles[:-1]),
                          ['勝率', '場均進球', '防守（反向）', '淨勝球', 'FIFA積分'])
        ax.set_ylim(0, 1)
        ax.set_title('球隊風格分群 — 雷達圖（k=%d，Silhouette=%.2f）' % (best_k, sil[best_k]),
                     fontsize=13, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, '14_cluster_radar.png'), dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  📊 figures/14_cluster_radar.png 存好了")

        # PCA 散點
        fig, ax = plt.subplots(figsize=(11, 7))
        for c in range(best_k):
            sub = df[df['cluster'] == c]
            ax.scatter(sub['pca1'], sub['pca2'], s=120, alpha=0.7,
                       label=cluster_names[c].replace('🔥','').replace('⚔️','').replace('🛡️','').replace('🌱','').strip(),
                       color=colors[c % len(colors)],
                       edgecolors='white', linewidths=1.2)
            for t, row in sub.iterrows():
                ax.annotate(t, (row['pca1'], row['pca2']), fontsize=8, alpha=0.85,
                            xytext=(5, 5), textcoords='offset points')
        ax.set_xlabel(f'主成分1（解釋變異 {pca.explained_variance_ratio_[0]*100:.1f}%）')
        ax.set_ylabel(f'主成分2（解釋變異 {pca.explained_variance_ratio_[1]*100:.1f}%）')
        ax.set_title('球隊風格 PCA 二維投影（48支參賽隊）', fontsize=13)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, '15_cluster_pca.png'), dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  📊 figures/15_cluster_pca.png 存好了")

    # 印出各群成員
    print("\n  分群結果：")
    for c in range(best_k):
        members = df[df['cluster'] == c].index.tolist()
        print(f"  {cluster_names[c]}（{len(members)} 隊）")
        print(f"    {', '.join(members)}")

    return out


# ============================================================
# 4) 評估視覺化（混淆矩陣 / ROC / Calibration）
# ============================================================
def eval_visualizations(match_df, fifa_df, clf, feat_cols):
    print("\n[4/4] 評估視覺化")
    # 用 2022 WC 當 hold-out test
    val = make_wc_dataset(match_df, fifa_df, 2022)
    if len(val) < 5:
        print("  ⚠️ 2022 WC 資料不足，略過評估視覺化")
        return
    Xv = pd.DataFrame([s['feat'] for s in val])[feat_cols]
    yv = np.array([s['label'] for s in val])
    yp = clf.predict(Xv)
    yp_proba = clf.predict_proba(Xv)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import (confusion_matrix, classification_report,
                                     roc_curve, auc, brier_score_loss)
    except Exception:
        return

    # 1) Confusion matrix
    cm = confusion_matrix(yv, yp)
    labels_name = ['Team1 Lose', 'Draw', 'Team1 Win']
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Reds')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels_name, rotation=15); ax.set_yticklabels(labels_name)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — 2022 WC Group Stage (acc={accuracy_score(yv, yp):.2f})')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '16_confusion_matrix_v2.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("  📊 figures/16_confusion_matrix_v2.png 存好了")

    # 2) ROC One-vs-Rest
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(yv, classes=[0,1,2])
    fig, ax = plt.subplots(figsize=(8, 6))
    for c, name in enumerate(labels_name):
        if y_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], yp_proba[:, c])
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')
    ax.plot([0,1], [0,1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — One-vs-Rest (2022 WC Test Set)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '17_roc_curves.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("  📊 figures/17_roc_curves.png 存好了")

    # 3) Feature importance
    fig, ax = plt.subplots(figsize=(8, 6))
    imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values()
    ax.barh(imp.index, imp.values, color='#e94560')
    ax.set_title('Feature Importance (XGBoost)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '18_feature_importance_v2.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("  📊 figures/18_feature_importance_v2.png 存好了")

    # 4) Calibration curve（"Win" class，index=2）
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(
        (yv == 2).astype(int), yp_proba[:, 2], n_bins=5, strategy='uniform'
    )

    # 5) 把所有原始數值存成 pkl，讓 app.py 能畫互動式 Plotly 圖
    roc_data = {}
    from sklearn.preprocessing import label_binarize
    y_bin_save = label_binarize(yv, classes=[0, 1, 2])
    for c, name in enumerate(labels_name):
        if y_bin_save[:, c].sum() == 0:
            continue
        fpr_c, tpr_c, _ = roc_curve(y_bin_save[:, c], yp_proba[:, c])
        roc_data[name] = {
            'fpr': fpr_c.tolist(),
            'tpr': tpr_c.tolist(),
            'auc': float(auc(fpr_c, tpr_c)),
        }

    eval_metrics = {
        'cm': cm.tolist(),
        'labels_name': labels_name,
        'accuracy': float(accuracy_score(yv, yp)),
        'roc': roc_data,
        'calibration': {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
        },
        'feature_importance': {
            'features': imp.index.tolist(),
            'values': imp.values.tolist(),
        },
        'n_test': int(len(yv)),
    }
    eval_path = os.path.join(MODEL_DIR, 'eval_metrics.pkl')
    with open(eval_path, 'wb') as f:
        pickle.dump(eval_metrics, f)
    print(f"  ✅ models/eval_metrics.pkl 存好了（test n={len(yv)}, acc={eval_metrics['accuracy']:.3f}）")


if __name__ == '__main__':
    print("="*60)
    print("世界盃 ML 2026 — Pretrain Script")
    print("="*60)
    match_df, fifa_df = load_data()
    clf, p1, p2, fc, va = train_models(match_df, fifa_df)
    eval_visualizations(match_df, fifa_df, clf, fc)
    cluster_out = run_clustering(match_df)
    mc = run_monte_carlo(match_df, fifa_df, clf, p1, p2, fc, n_sims=10000)
    print("\n" + "="*60)
    print("✅ 全部完成！models/ 已更新")

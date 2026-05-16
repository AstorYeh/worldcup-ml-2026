"""
🏆 Pretrain Script — 世界盃 ML v2.1
=====================================
把 app.py 內 walk-forward 訓練 + Monte Carlo + 新增的球隊風格分群
全部 offline 跑一遍，產出 pkl 檔放在 models/。

執行：python pretrain.py
產出：
    models/clf.pkl              XGBoost 分類器
    models/poisson1.pkl         主隊進球 Poisson
    models/poisson2.pkl         客隊進球 Poisson
    models/feat_cols.pkl        特徵欄位
    models/val_accs.pkl         Walk-Forward 驗證準確率
    models/mc_results.pkl       Monte Carlo 10,000 次奪冠機率
    models/team_clusters.pkl    球隊風格分群（KMeans）
    models/cluster_profiles.pkl 各群中心與描述
    figures/cluster_radar.png   雷達圖（4 群）
    figures/cluster_tsne.png    t-SNE 視覺化
    data_cache/match_df.pkl     比賽資料快取
    data_cache/fifa_df.pkl      FIFA 排名快取
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
# 2026 分組（與 app.py 同步）
# ============================================================
WC_2026_GROUPS = {
    'A': ['Mexico', 'Czechia', 'South Korea', 'South Africa'],
    'B': ['Canada', 'Switzerland', 'Qatar', 'Bosnia and Herzegovina'],
    'C': ['Brazil', 'Morocco', 'Scotland', 'Haiti'],
    'D': ['USA', 'Paraguay', 'Australia', 'Turkiye'],
    'E': ['Germany', 'Ecuador', 'Ivory Coast', 'Curacao'],
    'F': ['Netherlands', 'Japan', 'Tunisia', 'Sweden'],
    'G': ['Belgium', 'Iran', 'Egypt', 'New Zealand'],
    'H': ['Spain', 'Uruguay', 'Saudi Arabia', 'Cape Verde'],
    'I': ['France', 'Senegal', 'Norway', 'Iraq'],
    'J': ['Argentina', 'Austria', 'Algeria', 'Jordan'],
    'K': ['Portugal', 'Colombia', 'Uzbekistan', 'DR Congo'],
    'L': ['England', 'Croatia', 'Panama', 'Ghana'],
}

# 簡化版 FIFA 積分（從 app.py TEAM_INFO 提取）
TEAM_PTS = {
    'Mexico': 1701, 'Czechia': 1402, 'South Korea': 1569, 'South Africa': 1385,
    'Canada': 1502, 'Switzerland': 1712, 'Qatar': 1482, 'Bosnia and Herzegovina': 1402,
    'Brazil': 1819, 'Morocco': 1735, 'Scotland': 1390, 'Haiti': 1279,
    'USA': 1635, 'Paraguay': 1424, 'Australia': 1544, 'Turkiye': 1623,
    'Germany': 1692, 'Ecuador': 1535, 'Ivory Coast': 1448, 'Curacao': 1293,
    'Netherlands': 1760, 'Japan': 1640, 'Tunisia': 1505, 'Sweden': 1518,
    'Belgium': 1768, 'Iran': 1623, 'Egypt': 1516, 'New Zealand': 1247,
    'Spain': 1876, 'Uruguay': 1701, 'Saudi Arabia': 1433, 'Cape Verde': 1265,
    'France': 1877, 'Senegal': 1621, 'Norway': 1472, 'Iraq': 1436,
    'Argentina': 1874, 'Austria': 1580, 'Algeria': 1486, 'Jordan': 1378,
    'Portugal': 1752, 'Colombia': 1739, 'Uzbekistan': 1414, 'DR Congo': 1359,
    'England': 1825, 'Croatia': 1700, 'Panama': 1503, 'Ghana': 1360,
}
TEAM_RANK = {
    'Mexico': 15, 'Czechia': 50, 'South Korea': 24, 'South Africa': 59,
    'Canada': 38, 'Switzerland': 14, 'Qatar': 44, 'Bosnia and Herzegovina': 50,
    'Brazil': 5, 'Morocco': 11, 'Scotland': 52, 'Haiti': 86,
    'USA': 17, 'Paraguay': 57, 'Australia': 25, 'Turkiye': 18,
    'Germany': 13, 'Ecuador': 27, 'Ivory Coast': 39, 'Curacao': 79,
    'Netherlands': 7, 'Japan': 16, 'Tunisia': 36, 'Sweden': 33,
    'Belgium': 6, 'Iran': 19, 'Egypt': 31, 'New Zealand': 95,
    'Spain': 2, 'Uruguay': 11, 'Saudi Arabia': 56, 'Cape Verde': 88,
    'France': 1, 'Senegal': 21, 'Norway': 47, 'Iraq': 55,
    'Argentina': 3, 'Austria': 22, 'Algeria': 41, 'Jordan': 68,
    'Portugal': 8, 'Colombia': 9, 'Uzbekistan': 59, 'DR Congo': 61,
    'England': 4, 'Croatia': 12, 'Panama': 37, 'Ghana': 70,
}

CONFED_BONUS = {'UEFA': 0.08, 'CONMEBOL': 0.10, 'CONCACAF': 0.02, 'AFC': 0.01, 'CAF': 0.00, 'OFC': 0.00}
CONFED_MAP = {
    'France': 'UEFA','Spain': 'UEFA','England': 'UEFA','Germany': 'UEFA','Portugal': 'UEFA',
    'Netherlands': 'UEFA','Italy': 'UEFA','Belgium': 'UEFA','Croatia': 'UEFA','Switzerland': 'UEFA',
    'Poland': 'UEFA','Sweden': 'UEFA','Austria': 'UEFA','Czechia': 'UEFA','Scotland': 'UEFA',
    'Norway': 'UEFA','Bosnia and Herzegovina': 'UEFA','Turkiye': 'UEFA',
    'Brazil': 'CONMEBOL','Argentina': 'CONMEBOL','Uruguay': 'CONMEBOL','Colombia': 'CONMEBOL','Paraguay': 'CONMEBOL','Ecuador': 'CONMEBOL',
    'USA': 'CONCACAF','Mexico': 'CONCACAF','Canada': 'CONCACAF','Panama': 'CONCACAF','Curacao': 'CONCACAF','Haiti': 'CONCACAF',
    'Japan': 'AFC','South Korea': 'AFC','Iran': 'AFC','Australia': 'AFC','Saudi Arabia': 'AFC',
    'Iraq': 'AFC','Jordan': 'AFC','Uzbekistan': 'AFC','Qatar': 'AFC',
    'New Zealand': 'OFC',
    'Morocco': 'CAF','Egypt': 'CAF','Senegal': 'CAF','Algeria': 'CAF','Ghana': 'CAF',
    'Ivory Coast': 'CAF','Tunisia': 'CAF','Cape Verde': 'CAF','DR Congo': 'CAF','South Africa': 'CAF',
}

ALL_TEAMS = sorted({t for teams in WC_2026_GROUPS.values() for t in teams})


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

    print(f"[Data] match_df = {len(match_df):,} 場；fifa_df = {len(fifa_df):,} 筆排名")
    return match_df, fifa_df


# ============================================================
# FEATURE ENGINEERING（與 app.py 一致）
# ============================================================
def compute_team_strength(df, team, year, years_back=8):
    start_year = year - years_back
    tm = df[((df['home_team'] == team) | (df['away_team'] == team)) &
            (df['year'] >= start_year) & (df['year'] < year)].copy()
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


def compute_recent_form(df, team, year):
    recent = df[((df['home_team'] == team) | (df['away_team'] == team)) &
                (df['year'] >= year - 2) & (df['year'] < year)]
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
                       strength_cache=None, form_cache=None, ko_cache=None):
    """支援快取版本，加速大量訓練"""
    if strength_cache is None or (team1, year) not in strength_cache:
        s1 = compute_team_strength(match_df, team1, year)
        if strength_cache is not None:
            strength_cache[(team1, year)] = s1
    else:
        s1 = strength_cache[(team1, year)]

    if strength_cache is None or (team2, year) not in strength_cache:
        s2 = compute_team_strength(match_df, team2, year)
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
        f1 = compute_recent_form(match_df, team1, year)
        if form_cache is not None: form_cache[(team1, year)] = f1
    else:
        f1 = form_cache[(team1, year)]
    if form_cache is None or (team2, year) not in form_cache:
        f2 = compute_recent_form(match_df, team2, year)
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
    # 預先算 2026 所有可能對戰的勝率與進球期望
    strength_cache, form_cache, ko_cache = {}, {}, {}
    pairs = {}
    for g, teams in WC_2026_GROUPS.items():
        for i, t1 in enumerate(teams):
            for t2 in teams[i+1:]:
                feat = create_features_v2(t1, t2, 2026, match_df, fifa_df,
                                          strength_cache, form_cache, ko_cache)
                X = pd.DataFrame([feat])[feat_cols]
                prob = clf.predict_proba(X)[0]  # [L, D, W from team1]
                lam1 = max(0.1, float(poisson1.predict(X)[0]))
                lam2 = max(0.1, float(poisson2.predict(X)[0]))
                pairs[(t1, t2)] = (prob, lam1, lam2)
                pairs[(t2, t1)] = (prob[[2,1,0]], lam2, lam1)

    # 跨組對戰也需要：淘汰賽
    for ta in ALL_TEAMS:
        for tb in ALL_TEAMS:
            if ta == tb or (ta, tb) in pairs:
                continue
            try:
                feat = create_features_v2(ta, tb, 2026, match_df, fifa_df,
                                          strength_cache, form_cache, ko_cache)
                X = pd.DataFrame([feat])[feat_cols]
                prob = clf.predict_proba(X)[0]
                lam1 = max(0.1, float(poisson1.predict(X)[0]))
                lam2 = max(0.1, float(poisson2.predict(X)[0]))
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

    champion_count = {t: 0 for t in ALL_TEAMS}
    final_count    = {t: 0 for t in ALL_TEAMS}
    semi_count     = {t: 0 for t in ALL_TEAMS}
    quarter_count  = {t: 0 for t in ALL_TEAMS}
    r16_count      = {t: 0 for t in ALL_TEAMS}
    advance_count  = {t: 0 for t in ALL_TEAMS}  # 出小組

    for sim in range(n_sims):
        if sim % 1000 == 0:
            print(f"  sim {sim}/{n_sims}")
        # 小組賽
        group_results = {}
        for g, teams in WC_2026_GROUPS.items():
            pts = {t: 0 for t in teams}
            gd = {t: 0 for t in teams}
            gf = {t: 0 for t in teams}
            for i, t1 in enumerate(teams):
                for t2 in teams[i+1:]:
                    w, g1, g2 = sim_match(t1, t2)
                    gf[t1] += g1; gf[t2] += g2
                    gd[t1] += g1-g2; gd[t2] += g2-g1
                    if w == t1: pts[t1] += 3
                    elif w == t2: pts[t2] += 3
                    else: pts[t1] += 1; pts[t2] += 1
            # 排序
            standing = sorted(teams, key=lambda t: (-pts[t], -gd[t], -gf[t], rng.random()))
            group_results[g] = standing
            for t in standing[:2]:
                advance_count[t] += 1

        # 取前 2 + 最佳 8 個第 3 名 = 32 隊
        thirds = []
        for g, standing in group_results.items():
            t3 = standing[2]
            # 用點數做排序代理
            score = TEAM_PTS.get(t3, 1500)
            thirds.append((t3, score))
        thirds.sort(key=lambda x: -x[1])
        best_thirds = [t for t, _ in thirds[:8]]

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

    mc = {t: {
            'champ_pct': champion_count[t] / n_sims * 100,
            'final_pct': final_count[t] / n_sims * 100,
            'semi_pct': semi_count[t] / n_sims * 100,
            'quarter_pct': quarter_count[t] / n_sims * 100,
            'r16_pct': r16_count[t] / n_sims * 100,
            'win_pct': advance_count[t] / n_sims * 100,  # 出小組
          } for t in ALL_TEAMS}

    with open(os.path.join(MODEL_DIR, 'mc_results.pkl'), 'wb') as f:
        pickle.dump(mc, f)
    top10 = sorted(mc.items(), key=lambda x: -x[1]['champ_pct'])[:10]
    print("  Top 10 奪冠機率：")
    for t, d in top10:
        print(f"    {t:25s}  champ={d['champ_pct']:5.2f}%  final={d['final_pct']:5.2f}%  win_grp={d['win_pct']:5.1f}%")
    print(f"  💾 models/mc_results.pkl 存好了")
    return mc


# ============================================================
# 3) 球隊風格分群（KMeans + PCA）
# ============================================================
def run_clustering(match_df):
    print("\n[3/4] 球隊風格分群（KMeans, k=4）")
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

    # 嘗試多個 k 看 silhouette
    from sklearn.metrics import silhouette_score
    sil = {}
    for k in [3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        sil[k] = silhouette_score(Xs, labels)
        print(f"  k={k} silhouette = {sil[k]:.3f}")
    best_k = max(sil, key=sil.get)
    print(f"  最佳 k = {best_k}")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    df['cluster'] = labels

    # 為各群定名稱（看哪一群的進攻/防守/實力最高）
    centers_unscaled = scaler.inverse_transform(km.cluster_centers_)
    centers_df = pd.DataFrame(centers_unscaled, columns=feat_cols)

    # 根據 fifa_pts 排序，給 cluster 標籤
    cluster_rank = centers_df['fifa_pts'].rank(ascending=False).astype(int) - 1
    # cluster_names：中文版（供 Streamlit HTML 渲染）
    # cluster_names_en：純 ASCII（供 matplotlib 圖例，避免 CJK 亂碼）
    cluster_names = {}
    cluster_names_en = {}
    for c in range(best_k):
        rank = cluster_rank[c]
        if rank == 0:
            cluster_names[c]    = '🔥 強權型（高實力、高攻擊）'
            cluster_names_en[c] = 'Tier 1 - Elite (High Power)'
        elif rank == 1:
            cluster_names[c]    = '⚔️ 競爭型（中上實力、攻守平衡）'
            cluster_names_en[c] = 'Tier 2 - Competitive (Balanced)'
        elif rank == 2:
            cluster_names[c]    = '🛡️ 穩固型（中等實力、防守為主）'
            cluster_names_en[c] = 'Tier 3 - Defensive (Mid Tier)'
        elif rank == 3:
            cluster_names[c]    = '🌱 黑馬型（中下實力、潛力新銳）'
            cluster_names_en[c] = 'Tier 4 - Underdog (Emerging)'
        else:
            cluster_names[c]    = f'第 {rank+1} 群'
            cluster_names_en[c] = f'Tier {rank+1}'

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
    mc = run_monte_carlo(match_df, fifa_df, clf, p1, p2, fc, n_sims=5000)
    print("\n" + "="*60)
    print("✅ 全部完成！models/ 已更新")

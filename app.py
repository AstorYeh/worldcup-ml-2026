"""
🏆 2026 世界盃 ML 勝率分析與比分預測系統 v2.1
World Cup 2026 — ML Win Probability & Score Prediction
============================================================
真實資料：49,328場國際賽事 + 67,894筆FIFA排名
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="2026 世界盃 ML 預測系統",
    page_icon="⚽",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
/* ── Theme: 熱血體育風 ── */
:root {
    --bg-primary:   #0a0a0f;
    --bg-card:       #12121a;
    --bg-card2:      #1a1a2e;
    --accent-red:    #e94560;
    --accent-blue:   #0f3460;
    --accent-gold:   #f7c59f;
    --accent-cyan:   #00d4ff;
    --text-primary:  #f0f0f0;
    --text-muted:    #8899aa;
    --border:        rgba(233,69,96,0.18);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    color: white;
    margin: 8px 0;
    border: 1px solid rgba(233,69,96,0.15);
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
}
.metric-card h2 { margin: 0; font-size: 2.2rem; color: #00d4ff; }
.metric-card p { margin: 4px 0 0; color: #8899aa; font-size: 0.85rem; }

/* ── 小組卡片 ── */
.group-card {
    background: linear-gradient(145deg, #12121a 0%, #1a1a2e 100%);
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 16px;
    border: 1px solid rgba(233,69,96,0.12);
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
}
.group-header {
    background: linear-gradient(90deg, #e94560 0%, #c23a52 100%);
    padding: 8px 16px;
    font-size: 0.95rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.team-row {
    display: flex;
    align-items: center;
    padding: 8px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: #e0e0e0;
    font-size: 0.90rem;
    transition: background 0.15s;
}
.team-row:last-child { border-bottom: none; }
.team-row:hover { background: rgba(233,69,96,0.06); }
.team-flag { font-size: 1.3rem; margin-right: 10px; flex-shrink: 0; }
.team-cn { font-weight: 600; color: #f0f0f0; margin-right: 6px; min-width: 60px; }
.team-en { color: #8899aa; flex: 1; font-size: 0.82rem; }
.team-rank {
    font-size: 0.72rem;
    color: #00d4ff;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    padding: 2px 8px;
    border-radius: 20px;
    flex-shrink: 0;
}

/* Progress bar 勝率 */
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #e94560, #00d4ff);
}

/* 分頁分隔線 */
section[data-testid="stSidebarNav"] + div {
    border-left: 1px solid rgba(233,69,96,0.15);
}

/* 預測卡片 */
.pred-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 48 支球隊：國旗 + 中文名 + 英文名對照表
# ============================================================
TEAM_INFO = {
    # FIFA 世界排名（2026年4月 / 資料來源：FIFA官網）
    # === GROUP A ===
    'Mexico':         {'flag': '🇲🇽', 'iso': 'mx', 'cn': '墨西哥',         'en': 'Mexico',          'fifa_rank': 15, 'fifa_pts': 1701},
    'Czechia':        {'flag': '🇨🇿', 'iso': 'cz', 'cn': '捷克',           'en': 'Czechia',         'fifa_rank': 50, 'fifa_pts': 1402},
    'South Korea':    {'flag': '🇰🇷', 'iso': 'kr', 'cn': '南韓',           'en': 'South Korea',     'fifa_rank': 24, 'fifa_pts': 1569},
    'South Africa':   {'flag': '🇿🇦', 'iso': 'za', 'cn': '南非',           'en': 'South Africa',    'fifa_rank': 59, 'fifa_pts': 1385},
    # === GROUP B ===
    'Canada':         {'flag': '🇨🇦', 'iso': 'ca', 'cn': '加拿大',         'en': 'Canada',          'fifa_rank': 38, 'fifa_pts': 1502},
    'Switzerland':    {'flag': '🇨🇭', 'iso': 'ch', 'cn': '瑞士',           'en': 'Switzerland',     'fifa_rank': 14, 'fifa_pts': 1712},
    'Qatar':          {'flag': '🇶🇦', 'iso': 'qa', 'cn': '卡達',           'en': 'Qatar',           'fifa_rank': 44, 'fifa_pts': 1482},
    'Bosnia and Herzegovina': {'flag': '🇧🇦', 'iso': 'ba', 'cn': '波赫',   'en': 'Bosnia',          'fifa_rank': 50, 'fifa_pts': 1402},
    # === GROUP C ===
    'Brazil':         {'flag': '🇧🇷', 'iso': 'br', 'cn': '巴西',           'en': 'Brazil',          'fifa_rank': 5,  'fifa_pts': 1819},
    'Morocco':        {'flag': '🇲🇦', 'iso': 'ma', 'cn': '摩洛哥',         'en': 'Morocco',         'fifa_rank': 11, 'fifa_pts': 1735},
    'Scotland':       {'flag': '🏴󠁧󠁢󠁳󠁣󠁴󠁿', 'iso': 'gb-sct', 'cn': '蘇格蘭', 'en': 'Scotland',      'fifa_rank': 52, 'fifa_pts': 1390},
    'Haiti':          {'flag': '🇭🇹', 'iso': 'ht', 'cn': '海地',           'en': 'Haiti',           'fifa_rank': 86, 'fifa_pts': 1279},
    # === GROUP D ===
    'USA':            {'flag': '🇺🇸', 'iso': 'us', 'cn': '美國',           'en': 'USA',             'fifa_rank': 17, 'fifa_pts': 1635},
    'Paraguay':       {'flag': '🇵🇾', 'iso': 'py', 'cn': '巴拉圭',         'en': 'Paraguay',        'fifa_rank': 57, 'fifa_pts': 1424},
    'Australia':      {'flag': '🇦🇺', 'iso': 'au', 'cn': '澳洲',           'en': 'Australia',       'fifa_rank': 25, 'fifa_pts': 1544},
    'Turkiye':        {'flag': '🇹🇷', 'iso': 'tr', 'cn': '土耳其',         'en': 'Turkiye',         'fifa_rank': 18, 'fifa_pts': 1623},
    # === GROUP E ===
    'Germany':        {'flag': '🇩🇪', 'iso': 'de', 'cn': '德國',           'en': 'Germany',         'fifa_rank': 13, 'fifa_pts': 1692},
    'Ecuador':        {'flag': '🇪🇨', 'iso': 'ec', 'cn': '厄瓜多',         'en': 'Ecuador',         'fifa_rank': 27, 'fifa_pts': 1535},
    'Ivory Coast':    {'flag': '🇨🇮', 'iso': 'ci', 'cn': '象牙海岸',       'en': 'Ivory Coast',     'fifa_rank': 39, 'fifa_pts': 1448},
    'Curacao':        {'flag': '🇨🇼', 'iso': 'cw', 'cn': '庫拉索',         'en': 'Curacao',         'fifa_rank': 79, 'fifa_pts': 1293},
    # === GROUP F ===
    'Netherlands':    {'flag': '🇳🇱', 'iso': 'nl', 'cn': '荷蘭',           'en': 'Netherlands',     'fifa_rank': 7,  'fifa_pts': 1760},
    'Japan':          {'flag': '🇯🇵', 'iso': 'jp', 'cn': '日本',           'en': 'Japan',           'fifa_rank': 16, 'fifa_pts': 1640},
    'Tunisia':        {'flag': '🇹🇳', 'iso': 'tn', 'cn': '突尼斯',         'en': 'Tunisia',         'fifa_rank': 36, 'fifa_pts': 1505},
    'Sweden':         {'flag': '🇸🇪', 'iso': 'se', 'cn': '瑞典',           'en': 'Sweden',          'fifa_rank': 33, 'fifa_pts': 1518},
    # === GROUP G ===
    'Belgium':        {'flag': '🇧🇪', 'iso': 'be', 'cn': '比利時',         'en': 'Belgium',         'fifa_rank': 6,  'fifa_pts': 1768},
    'Iran':           {'flag': '🇮🇷', 'iso': 'ir', 'cn': '伊朗',           'en': 'Iran',            'fifa_rank': 19, 'fifa_pts': 1623},
    'Egypt':          {'flag': '🇪🇬', 'iso': 'eg', 'cn': '埃及',           'en': 'Egypt',           'fifa_rank': 31, 'fifa_pts': 1516},
    'New Zealand':    {'flag': '🇳🇿', 'iso': 'nz', 'cn': '紐西蘭',         'en': 'New Zealand',     'fifa_rank': 95, 'fifa_pts': 1247},
    # === GROUP H ===
    'Spain':          {'flag': '🇪🇸', 'iso': 'es', 'cn': '西班牙',         'en': 'Spain',           'fifa_rank': 2,  'fifa_pts': 1876},
    'Uruguay':        {'flag': '🇺🇾', 'iso': 'uy', 'cn': '烏拉圭',         'en': 'Uruguay',         'fifa_rank': 11, 'fifa_pts': 1701},
    'Saudi Arabia':   {'flag': '🇸🇦', 'iso': 'sa', 'cn': '沙烏地阿拉伯',   'en': 'Saudi Arabia',    'fifa_rank': 56, 'fifa_pts': 1433},
    'Cape Verde':     {'flag': '🇨🇻', 'iso': 'cv', 'cn': '維德角',         'en': 'Cape Verde',      'fifa_rank': 88, 'fifa_pts': 1265},
    # === GROUP I ===
    'France':         {'flag': '🇫🇷', 'iso': 'fr', 'cn': '法國',           'en': 'France',          'fifa_rank': 1,  'fifa_pts': 1877},
    'Senegal':        {'flag': '🇸🇳', 'iso': 'sn', 'cn': '塞內加爾',       'en': 'Senegal',         'fifa_rank': 21, 'fifa_pts': 1621},
    'Norway':         {'flag': '🇳🇴', 'iso': 'no', 'cn': '挪威',           'en': 'Norway',          'fifa_rank': 47, 'fifa_pts': 1472},
    'Iraq':           {'flag': '🇮🇶', 'iso': 'iq', 'cn': '伊拉克',         'en': 'Iraq',            'fifa_rank': 55, 'fifa_pts': 1436},
    # === GROUP J ===
    'Argentina':      {'flag': '🇦🇷', 'iso': 'ar', 'cn': '阿根廷',         'en': 'Argentina',       'fifa_rank': 3,  'fifa_pts': 1874},
    'Austria':        {'flag': '🇦🇹', 'iso': 'at', 'cn': '奧地利',         'en': 'Austria',         'fifa_rank': 22, 'fifa_pts': 1580},
    'Algeria':        {'flag': '🇩🇿', 'iso': 'dz', 'cn': '阿爾及利亞',     'en': 'Algeria',         'fifa_rank': 41, 'fifa_pts': 1486},
    'Jordan':         {'flag': '🇯🇴', 'iso': 'jo', 'cn': '約旦',           'en': 'Jordan',          'fifa_rank': 68, 'fifa_pts': 1378},
    # === GROUP K ===
    'Portugal':       {'flag': '🇵🇹', 'iso': 'pt', 'cn': '葡萄牙',         'en': 'Portugal',        'fifa_rank': 8,  'fifa_pts': 1752},
    'Colombia':       {'flag': '🇨🇴', 'iso': 'co', 'cn': '哥倫比亞',       'en': 'Colombia',        'fifa_rank': 9,  'fifa_pts': 1739},
    'Uzbekistan':     {'flag': '🇺🇿', 'iso': 'uz', 'cn': '烏茲別克',       'en': 'Uzbekistan',      'fifa_rank': 59, 'fifa_pts': 1414},
    'DR Congo':       {'flag': '🇨🇩', 'iso': 'cd', 'cn': '民主剛果',       'en': 'DR Congo',        'fifa_rank': 61, 'fifa_pts': 1359},
    # === GROUP L ===
    'England':        {'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'iso': 'gb-eng', 'cn': '英格蘭', 'en': 'England',      'fifa_rank': 4,  'fifa_pts': 1825},
    'Croatia':        {'flag': '🇭🇷', 'iso': 'hr', 'cn': '克羅埃西亞',     'en': 'Croatia',         'fifa_rank': 12, 'fifa_pts': 1700},
    'Panama':         {'flag': '🇵🇦', 'iso': 'pa', 'cn': '巴拿馬',         'en': 'Panama',          'fifa_rank': 37, 'fifa_pts': 1503},
    'Ghana':          {'flag': '🇬🇭', 'iso': 'gh', 'cn': '迦納',           'en': 'Ghana',           'fifa_rank': 70, 'fifa_pts': 1360},
}


def team_display(name):
    """回傳：🇦🇷 阿根廷 (Argentina) - 世界排名第 1 名"""
    info = TEAM_INFO.get(name, {'flag': '🏳️', 'cn': name, 'en': name, 'fifa_rank': 99, 'fifa_pts': 0})
    return f"{info['flag']} {info['cn']} ({info['en']}) - 世界排名第 {info['fifa_rank']} 名", info

def team_pts(name):
    """取得球隊 FIFA 積分"""
    return TEAM_INFO.get(name, {}).get('fifa_pts', 1500)

def team_rank(name):
    """取得球隊 FIFA 世界排名"""
    return TEAM_INFO.get(name, {}).get('fifa_rank', 99)

# ============================================================
# 2026 世界盃分組
# ============================================================
WC_2026_GROUPS = {
    # 2026 FIFA 世界盃正式分組（2025/12/5 抽籤 + 2026年3-4月附加賽確認）
    # 資料來源：NBC Sports / FOX Sports / FIFA 官方（April 2026）
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


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_match_data():
    """載入真實國際比賽（直接從URL讀取）"""
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df = df.dropna(subset=['home_score', 'away_score'])
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    return df

@st.cache_data
def load_fifa_ranking():
    """載入 FIFA 排名歷史（直接從URL讀取）"""
    url = "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/ranking_fifa_historical.csv"
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df['date'])
    return df

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def compute_team_strength(df, team, year, years_back=8):
    """計算球隊綜合實力（只看近 N 年，含時間衰減）"""
    start_year = year - years_back
    team_matches = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['year'] >= start_year) & (df['year'] < year)
    ].copy()

    if len(team_matches) == 0:
        return {'win_rate': 0.35, 'draw_rate': 0.25, 'avg_goals': 1.2, 'avg_conceded': 1.3, 'matches': 0}

    results = []
    for _, row in team_matches.iterrows():
        if row['home_team'] == team:
            gf, ga = row['home_score'], row['away_score']
        else:
            gf, ga = row['away_score'], row['home_score']

        years_ago = year - row['year']
        weight = np.exp(-0.1 * years_ago)  # 時間衰減

        if gf > ga:
            results.append({'result': 'W', 'gf': gf, 'ga': ga, 'weight': weight})
        elif gf == ga:
            results.append({'result': 'D', 'gf': gf, 'ga': ga, 'weight': weight})
        else:
            results.append({'result': 'L', 'gf': gf, 'ga': ga, 'weight': weight})

    if not results:
        return {'win_rate': 0.35, 'draw_rate': 0.25, 'avg_goals': 1.2, 'avg_conceded': 1.3, 'matches': 0}

    res_df = pd.DataFrame(results)
    total_weight = res_df['weight'].sum()

    win_rate = res_df[res_df['result'] == 'W']['weight'].sum() / total_weight
    draw_rate = res_df[res_df['result'] == 'D']['weight'].sum() / total_weight
    avg_goals = (res_df['gf'] * res_df['weight']).sum() / total_weight
    avg_conceded = (res_df['ga'] * res_df['weight']).sum() / total_weight

    return {
        'win_rate': win_rate,
        'draw_rate': draw_rate,
        'avg_goals': avg_goals,
        'avg_conceded': avg_conceded,
        'matches': len(team_matches)
    }

def compute_recent_form(df, team, year):
    """近2年國際賽加權勝率（近期狀態）"""
    recent = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['year'] >= year - 2) & (df['year'] < year)
    ]
    if len(recent) == 0:
        return 0.35
    wins = sum(1 for _, row in recent.iterrows() if
               (row['home_team'] == team and row['home_score'] > row['away_score']) or
               (row['away_team'] == team and row['away_score'] > row['home_score']))
    return wins / len(recent)

def compute_knockout_exp(df, team):
    """該球隊歷史世界盃淘汰賽場次（經驗值）"""
    wc_knockout = df[
        (df['tournament'].str.contains('World Cup', na=False)) &
        (~df['tournament'].str.contains('qualif', case=False, na=False)) &
        ((df['home_team'] == team) | (df['away_team'] == team))
    ]
    return len(wc_knockout)

# 足聯加成：UEFA/CONMEBOL 歷史勝率顯著較高
CONFED_BONUS = {
    'UEFA': 0.08, 'CONMEBOL': 0.10, 'CONCACAF': 0.02,
    'AFC': 0.01, 'CAF': 0.00, 'OFC': 0.00
}

CONFED_MAP = {
    # UEFA
    'France': 'UEFA', 'Spain': 'UEFA', 'England': 'UEFA', 'Germany': 'UEFA',
    'Portugal': 'UEFA', 'Netherlands': 'UEFA', 'Italy': 'UEFA', 'Belgium': 'UEFA',
    'Croatia': 'UEFA', 'Switzerland': 'UEFA', 'Poland': 'UEFA', 'Sweden': 'UEFA',
    'Austria': 'UEFA', 'Czechia': 'UEFA', 'Scotland': 'UEFA', 'Norway': 'UEFA',
    'Bosnia and Herzegovina': 'UEFA', 'Turkiye': 'UEFA',
    # CONMEBOL
    'Brazil': 'CONMEBOL', 'Argentina': 'CONMEBOL', 'Uruguay': 'CONMEBOL',
    'Colombia': 'CONMEBOL',
    # CONCACAF
    'USA': 'CONCACAF', 'Mexico': 'CONCACAF', 'Canada': 'CONCACAF',
    'Panama': 'CONCACAF', 'Curacao': 'CONCACAF',
    # AFC
    'Japan': 'AFC', 'South Korea': 'AFC', 'Iran': 'AFC', 'Australia': 'AFC',
    'Saudi Arabia': 'AFC', 'Iraq': 'AFC', 'Jordan': 'AFC', 'Uzbekistan': 'AFC',
    'Qatar': 'AFC', 'New Zealand': 'OFC',
    # CAF
    'Morocco': 'CAF', 'Egypt': 'CAF', 'Senegal': 'CAF', 'Algeria': 'CAF',
    'Ghana': 'CAF', 'Ivory Coast': 'CAF', 'Tunisia': 'CAF', 'Cape Verde': 'CAF',
    'DR Congo': 'CAF', 'South Africa': 'CAF',
}

# ============================================================
# FIFA RANKING HELPERS (historical)
# ============================================================
def get_historical_ranking(fifa_df, team, date):
    """取得某時間點最近的一筆 FIFA 排名"""
    row = fifa_df[(fifa_df['team'] == team) & (fifa_df['date'] <= date)].sort_values('date').tail(1)
    if len(row) == 0:
        return 1500, 50  # default pts, default rank
    return float(row.iloc[0]['total_points']), int(row.iloc[0]['id_num'])


# ============================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================
def create_features_v2(team1, team2, year, match_df, fifa_df):
    """為對戰建立 ML 特徵（使用歷史 FIFA 排名）"""
    s1 = compute_team_strength(match_df, team1, year)
    s2 = compute_team_strength(match_df, team2, year)

    # 歷史排名（該年世界盃前最近的一筆）
    pts1, _ = get_historical_ranking(fifa_df, team1, f"{year}-05-01")
    pts2, _ = get_historical_ranking(fifa_df, team2, f"{year}-05-01")

    r1 = team_pts(team1)
    r2 = team_pts(team2)

    confed1 = CONFED_MAP.get(team1, 'CAF')
    confed2 = CONFED_MAP.get(team2, 'CAF')

    return {
        'pts_diff': pts1 - pts2,           # 歷史 FIFA 積分差（主要）
        'rank_diff': r1 - r2,              # 2026 靜態排名差（輔助）
        'win_rate_diff': s1['win_rate'] - s2['win_rate'],
        'avg_goals_diff': s1['avg_goals'] - s2['avg_goals'],
        'defense_diff': s1['avg_conceded'] - s2['avg_conceded'],
        'form_diff': (s1['win_rate'] * s1['avg_goals']) - (s2['win_rate'] * s2['avg_goals']),
        'experience': s1['matches'] + s2['matches'],
        'goals_product': s1['avg_goals'] * s2['avg_goals'],
        'confed_bonus': CONFED_BONUS.get(confed1, 0) - CONFED_BONUS.get(confed2, 0),
        'recent_form': compute_recent_form(match_df, team1, year) - compute_recent_form(match_df, team2, year),
        'knockout_exp': compute_knockout_exp(match_df, team1) - compute_knockout_exp(match_df, team2),
    }


def _make_wc_dataset_v2(match_df, fifa_df, year):
    """取出某屆世界盃小組賽，並用歷史資料當特徵"""
    samples = []
    for g, teams in WC_2026_GROUPS.items():
        for i, t1 in enumerate(teams):
            for t2 in teams[i+1:]:
                try:
                    feat = create_features_v2(t1, t2, year, match_df, fifa_df)
                    m = match_df[
                        (((match_df['home_team']==t1)&(match_df['away_team']==t2)) |
                         ((match_df['home_team']==t2)&(match_df['away_team']==t1))) &
                        (match_df['year']==year) &
                        (match_df['tournament'].str.contains('World Cup', na=False))
                    ]
                    if len(m) == 0:
                        continue
                    row = m.iloc[0]
                    gf = row['home_score'] if row['home_team'] == t1 else row['away_score']
                    ga = row['away_score'] if row['home_team'] == t1 else row['home_score']
                    samples.append({
                        'feat': feat,
                        'label': 0 if gf < ga else (1 if gf == ga else 2),
                        'gf': gf, 'ga': ga,
                        'team1': t1, 'team2': t2,
                    })
                except:
                    continue
    return samples


def _make_all_intl_dataset(match_df, fifa_df, year, years_back=8):
    """用該年之前所有國際賽當訓練集（排除 World Cup 當屆）"""
    samples = []
    start_year = year - years_back
    # 所有國際賽（非 World Cup，避免資料洩漏）
    intl = match_df[
        (match_df['year'] >= start_year) &
        (match_df['year'] < year) &
        (~match_df['tournament'].str.contains('World Cup', na=False))
    ].copy()

    # 只取有在 2026 分組中的球隊
    all_teams = set()
    for teams in WC_2026_GROUPS.values():
        all_teams.update(teams)

    for _, row in intl.iterrows():
        t1, t2 = row['home_team'], row['away_team']
        if t1 not in all_teams or t2 not in all_teams:
            continue
        if row['year'] < 1990:  # 太久遠的比賽參考價值低
            continue
        try:
            feat = create_features_v2(t1, t2, row['year'], match_df, fifa_df)
            gf = row['home_score']
            ga = row['away_score']
            samples.append({
                'feat': feat,
                'label': 0 if gf < ga else (1 if gf == ga else 2),
                'gf': gf, 'ga': ga,
                'team1': t1, 'team2': t2,
            })
        except:
            continue
    return samples

# ============================================================
# MODEL TRAINING
# ============================================================
def _make_wc_dataset(match_df, year):
    """取出某屆世界盃的小組賽+資格賽結果當特徵樣本"""
    samples = []
    for g, teams in WC_2026_GROUPS.items():
        for i, t1 in enumerate(teams):
            for t2 in teams[i+1:]:
                try:
                    feat = create_features(t1, t2, year, match_df)
                    m = match_df[
                        (((match_df['home_team']==t1)&(match_df['away_team']==t2)) |
                         ((match_df['home_team']==t2)&(match_df['away_team']==t1))) &
                        (match_df['year']==year) &
                        (match_df['tournament'].str.contains('World Cup', na=False))
                    ]
                    if len(m) == 0:
                        continue
                    row = m.iloc[0]
                    gf = row['home_score'] if row['home_team'] == t1 else row['away_score']
                    ga = row['away_score'] if row['home_team'] == t1 else row['home_score']
                    samples.append({
                        'feat': feat,
                        'label': 0 if gf < ga else (1 if gf == ga else 2),  # 0=輸,1=平,2=贏
                        'gf': gf, 'ga': ga
                    })
                except:
                    continue
    return samples

# 保持舊名向後相容（UI 其他頁面用）
def create_features(team1, team2, year, match_df, fifa_df=None):
    """向後相容：沒有 fifa_df 時用 2026 靜態排名"""
    if fifa_df is not None:
        return create_features_v2(team1, team2, year, match_df, fifa_df)
    s1 = compute_team_strength(match_df, team1, year)
    s2 = compute_team_strength(match_df, team2, year)
    r1, r2 = team_pts(team1), team_pts(team2)
    confed1 = CONFED_MAP.get(team1, 'CAF')
    confed2 = CONFED_MAP.get(team2, 'CAF')
    return {
        'rank_diff': r1 - r2,
        'win_rate_diff': s1['win_rate'] - s2['win_rate'],
        'avg_goals_diff': s1['avg_goals'] - s2['avg_goals'],
        'defense_diff': s1['avg_conceded'] - s2['avg_conceded'],
        'form_diff': (s1['win_rate'] * s1['avg_goals']) - (s2['win_rate'] * s2['avg_goals']),
        'experience': s1['matches'] + s2['matches'],
        'goals_product': s1['avg_goals'] * s2['avg_goals'],
        'confed_bonus': CONFED_BONUS.get(confed1, 0) - CONFED_BONUS.get(confed2, 0),
        'recent_form': compute_recent_form(match_df, team1, year) - compute_recent_form(match_df, team2, year),
        'knockout_exp': compute_knockout_exp(match_df, team1) - compute_knockout_exp(match_df, team2),
    }

# ============================================================
# 預訓練 pkl 載入（v2.2 新增：解決 Streamlit Cloud 冷啟動延遲）
# ============================================================
@st.cache_resource
def load_pretrained():
    """嘗試從 models/ 載入預訓練模型；找不到回 None。"""
    import os as _os, pickle as _pickle
    base = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'models')
    paths = {
        'clf': _os.path.join(base, 'clf.pkl'),
        'poisson1': _os.path.join(base, 'poisson1.pkl'),
        'poisson2': _os.path.join(base, 'poisson2.pkl'),
        'feat_cols': _os.path.join(base, 'feat_cols.pkl'),
        'val_accs': _os.path.join(base, 'val_accs.pkl'),
    }
    if not all(_os.path.exists(p) for p in paths.values()):
        return None
    out = {}
    try:
        for k, p in paths.items():
            with open(p, 'rb') as _f:
                out[k] = _pickle.load(_f)
        return out
    except Exception:
        return None

@st.cache_resource
def load_mc_results():
    """嘗試從 models/mc_results.pkl 載入；找不到回 None。"""
    import os as _os, pickle as _pickle
    p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'models', 'mc_results.pkl')
    if not _os.path.exists(p):
        return None
    try:
        with open(p, 'rb') as _f:
            return _pickle.load(_f)
    except Exception:
        return None

@st.cache_resource
def load_team_clusters():
    """嘗試從 models/team_clusters.pkl 載入。"""
    import os as _os, pickle as _pickle
    p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'models', 'team_clusters.pkl')
    if not _os.path.exists(p):
        return None
    try:
        with open(p, 'rb') as _f:
            return _pickle.load(_f)
    except Exception:
        return None


@st.cache_resource
def load_eval_metrics():
    """嘗試從 models/eval_metrics.pkl 載入模型評估數據。"""
    import os as _os, pickle as _pickle
    p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'models', 'eval_metrics.pkl')
    if not _os.path.exists(p):
        return None
    try:
        with open(p, 'rb') as _f:
            return _pickle.load(_f)
    except Exception:
        return None


# ============================================================
# WALK-FORWARD MODEL TRAINING
# ============================================================
@st.cache_resource
def train_models_walkforward(match_df, fifa_df):
    """
    Walk-forward 訓練：
    - 用歷史國際賽（1990-2022，不含 WC）作為主要訓練集
    - 用 2010→2014→2018→2022 WC 小組賽做漸進驗證
    - 回傳 2026 預測用模型
    """
    from sklearn.linear_model import PoissonRegressor

    WC_YEARS = [2010, 2014, 2018, 2022]

    # ── Step 1: 用 1990-2006 國際賽，預測 2010 WC（小樣本 warm-up）──
    train_2010 = _make_all_intl_dataset(match_df, fifa_df, 2010, years_back=20)
    val_2010   = _make_wc_dataset_v2(match_df, fifa_df, 2010)

    # ── Step 2: 用 1990-2010 國際賽，預測 2014 WC ─
    train_2014 = _make_all_intl_dataset(match_df, fifa_df, 2014, years_back=24)
    val_2014   = _make_wc_dataset_v2(match_df, fifa_df, 2014)

    # ── Step 3: 用 1990-2014 國際賽，預測 2018 WC ─
    train_2018 = _make_all_intl_dataset(match_df, fifa_df, 2018, years_back=28)
    val_2018   = _make_wc_dataset_v2(match_df, fifa_df, 2018)

    # ── Step 4: 用 1990-2018 國際賽，預測 2022 WC ─
    train_2022 = _make_all_intl_dataset(match_df, fifa_df, 2022, years_back=32)
    val_2022   = _make_wc_dataset_v2(match_df, fifa_df, 2022)

    # ── 最終模型：用 1990-2022 國際賽，預測 2026 ─
    train_final = _make_all_intl_dataset(match_df, fifa_df, 2026, years_back=36)

    print(f"[Walk-Forward] 2010 train={len(train_2010)}, val={len(val_2010)}")
    print(f"[Walk-Forward] 2014 train={len(train_2014)}, val={len(val_2014)}")
    print(f"[Walk-Forward] 2018 train={len(train_2018)}, val={len(val_2018)}")
    print(f"[Walk-Forward] 2022 train={len(train_2022)}, val={len(val_2022)}")
    print(f"[Final] 2026 train={len(train_final)}")

    # 至少要有 200 樣本才訓練
    if len(train_final) < 200:
        return None, None, None, None, None, None

    X_tr = pd.DataFrame([s['feat'] for s in train_final])
    y_tr = np.array([s['label'] for s in train_final])
    feat_cols = list(X_tr.columns)

    # XGBoost
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, use_label_encoder=False, eval_metric='mlogloss'
    )
    clf.fit(X_tr, y_tr, verbose=False)

    # Poisson 迴歸
    y_g1 = np.array([s['gf'] for s in train_final])
    y_g2 = np.array([s['ga'] for s in train_final])
    poisson1 = PoissonRegressor(alpha=0.1, max_iter=500)
    poisson2 = PoissonRegressor(alpha=0.1, max_iter=500)
    poisson1.fit(X_tr, y_g1)
    poisson2.fit(X_tr, y_g2)

    # 計算驗證集 accuracy（每次 Walk-Forward）
    val_accs = {}
    for yr, val_data in [(2010, val_2010), (2014, val_2014), (2018, val_2018), (2022, val_2022)]:
        if len(val_data) >= 5:
            X_v = pd.DataFrame([s['feat'] for s in val_data])[feat_cols]
            y_v = np.array([s['label'] for s in val_data])
            # 只取特徵有的欄位
            val_accs[yr] = accuracy_score(y_v, clf.predict(X_v))
        else:
            val_accs[yr] = None

    # 2022 是測試集
    test_acc = val_accs.get(2022)
    avg_val_acc = np.mean([v for v in val_accs.values() if v is not None])

    return clf, avg_val_acc, test_acc, poisson1, poisson2, feat_cols, val_accs


# ============================================================
# PREDICTION
# ============================================================
def predict_match(team1, team2, year, match_df, fifa_df, clf, poisson1, poisson2, feat_cols):
    """預測單場（使用歷史 FIFA 排名）"""
    feat = create_features_v2(team1, team2, year, match_df, fifa_df)
    X = pd.DataFrame([feat])[feat_cols]

    goal1 = max(0, int(round(float(poisson1.predict(X)[0]))))
    goal2 = max(0, int(round(float(poisson2.predict(X)[0]))))

    proba = clf.predict_proba(X)
    classes = clf.classes_

    # 類別對照：0=輸( loss), 1=平( draw), 2=贏( win)
    prob_win = prob_draw = prob_loss = 0.3
    for c, p in zip(classes, proba[0]):
        if c == 2: prob_win = p
        elif c == 1: prob_draw = p
        else: prob_loss = p

    # Poisson 交叉修正
    if goal1 > goal2 + 1:
        prob_win = max(prob_win, 0.50)
    elif goal2 > goal1 + 1:
        prob_loss = max(prob_loss, 0.50)

    r1, r2 = team_pts(team1), team_pts(team2)
    rank1, rank2 = team_rank(team1), team_rank(team2)
    info1 = TEAM_INFO.get(team1, {'flag': '🏳️', 'cn': team1, 'en': team1})
    info2 = TEAM_INFO.get(team2, {'flag': '🏳️', 'cn': team2, 'en': team2})

    return {
        'team1_display': f"{info1['flag']} {info1['cn']} ({info1['en']}) - 世界排名第 {rank1} 名",
        'team2_display': f"{info2['flag']} {info2['cn']} ({info2['en']}) - 世界排名第 {rank2} 名",
        'team1': team1, 'team2': team2,
        'win_prob': float(prob_win),
        'draw_prob': float(prob_draw),
        'loss_prob': float(prob_loss),
        'goal1': goal1, 'goal2': goal2,
        'rank1': rank1, 'rank2': rank2,
    }

# ============================================================
# MONTE CARLO SIMULATION
# ============================================================
def monte_carlo(match_df, fifa_df, clf, poisson1, poisson2, feat_cols, n_sims=10000):
    """Monte Carlo 奪冠模擬 — 動態赔率 + XGBoost 淘汰賽"""
    np.random.seed(42)
    all_teams = [t for ts in WC_2026_GROUPS.values() for t in ts]
    win_count = {t: 0 for t in all_teams}
    champion_count = {t: 0 for t in all_teams}

    # UEFA/CONMEBOL 基礎勝率加成
    BASE_CONFED_BONUS = {
        'UEFA': 0.08, 'CONMEBOL': 0.10, 'CONCACAF': 0.02,
        'AFC': 0.01, 'CAF': 0.00, 'OFC': 0.00
    }

    def team_base_winrate(team):
        """球隊基礎勝率：FIFA排名轉換 + 足聯加成"""
        pts = team_pts(team)
        base = 1 / (1 + np.exp(-2.0 * (pts - 1500) / 800))
        confed = CONFED_MAP.get(team, 'CAF')
        return min(base * 0.85 + BASE_CONFED_BONUS.get(confed, 0), 0.92)

    def sim_group_match(t1, t2):
        """小組賽單場模擬：動態p_draw"""
        bw1 = team_base_winrate(t1)
        bw2 = team_base_winrate(t2)
        # 實力接近 → p_draw 更高（最高0.28），實力差距大 → 最低0.15
        rank_diff = abs(team_rank(t1) - team_rank(t2))
        p_draw = max(0.15, min(0.28, 0.28 - rank_diff * 0.005))
        p_win = bw1 / (bw1 + bw2) * (1 - p_draw)
        r = np.random.random()
        if r < p_win:
            return t1, 3, t2, 0
        elif r < p_win + p_draw:
            return t1, 1, t2, 1
        else:
            return t1, 0, t2, 3

    for _ in range(n_sims):
        # ── 小組賽（單次模擬） ──
        group_pts = {}
        for g, teams in WC_2026_GROUPS.items():
            pts = {t: 0 for t in teams}
            for i, t1 in enumerate(teams):
                for t2 in teams[i+1:]:
                    t1_pts, t2_pts = sim_group_match(t1, t2)
                    pts[t1] += t1_pts; pts[t2] += t2_pts
            sorted_pts = sorted(pts.items(), key=lambda x: x[1], reverse=True)
            for t, _ in sorted_pts[:2]:
                win_count[t] += 1  # 累計晉級次數

        # ── 16強 → 8強 → 4強 → 決賽（使用 XGBoost）──
        # 取小組賽積分前16名（單次模擬結果，非累積）
        sim_winners = sorted(win_count.items(), key=lambda x: x[1], reverse=True)[:16]
        bracket = [t for t, _ in sim_winners]
        np.random.shuffle(bracket)

        # Round of 16
        r16_winners = []
        for i in range(0, 16, 2):
            t1, t2 = bracket[i], bracket[i+1]
            feat = create_features(t1, t2, 2026, match_df)
            X = pd.DataFrame([feat])[feat_cols]
            proba = clf.predict_proba(X)[0]
            win_prob = max(proba)
            winner = t1 if np.random.random() < win_prob + 0.05 else t2
            r16_winners.append(winner)

        # Quarter finals
        qf_winners = []
        for i in range(0, 8, 2):
            t1, t2 = r16_winners[i], r16_winners[i+1]
            feat = create_features(t1, t2, 2026, match_df)
            X = pd.DataFrame([feat])[feat_cols]
            proba = clf.predict_proba(X)[0]
            win_prob = max(proba)
            winner = t1 if np.random.random() < win_prob + 0.05 else t2
            qf_winners.append(winner)

        # Semis
        sf_winners = []
        for i in range(0, 4, 2):
            t1, t2 = qf_winners[i], qf_winners[i+1]
            feat = create_features(t1, t2, 2026, match_df)
            X = pd.DataFrame([feat])[feat_cols]
            proba = clf.predict_proba(X)[0]
            win_prob = max(proba)
            winner = t1 if np.random.random() < win_prob + 0.05 else t2
            sf_winners.append(winner)

        # Final
        t1, t2 = sf_winners[0], sf_winners[1]
        feat = create_features(t1, t2, 2026, match_df)
        X = pd.DataFrame([feat])[feat_cols]
        proba = clf.predict_proba(X)[0]
        win_prob = max(proba)
        champion = t1 if np.random.random() < win_prob + 0.05 else t2
        champion_count[champion] += 1

    result = {t: {'win_pct': win_count[t] / n_sims * 100, 'champ_pct': champion_count[t] / n_sims * 100} for t in all_teams}
    return result

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🏆 世界盃 ML 導航")
page = st.sidebar.radio("選擇頁面", [
    "📊 專題總覽",
    "🔮 2026 預測",
    "📈 數據分析",
    "🎯 球隊風格分群",
    "🏅 奪冠預測",
    "📅 完整賽程",
])

# ============================================================
# PAGE 1: 專題總覽
# ============================================================
if page == "📊 專題總覽":
    st.title("⚽ 2026 世界盃 ML 勝率分析系統")
    st.markdown("**真實資料：49,328 場國際比賽 · 67,894 筆 FIFA 排名**")
    st.markdown("---")

    match_df = load_match_data()
    fifa_df = load_fifa_ranking()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h2>49,328</h2><p>歷史比賽場次</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>48</h2><p>參賽球隊</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2>235</h2><p>國家球隊數</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h2>12</h2><p>小組數</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📋 2026 世界盃小組分組（🇺🇸🇨🇦🇲🇽 主辦）")

    # 4列格狀佈局
    # 4列 CSS Grid：小組卡片，格子固定、內容靠上、文字自適應
    GRID_CSS = """
    <style>
    .grp-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 14px 0 20px;
    }
    .grp-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #0f3460 100%);
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.07);
        box-shadow: 0 4px 18px rgba(0,0,0,0.35);
        display: flex;
        flex-direction: column;
        height: 218px;           /* 固定高度 */
    }
    .grp-hdr {
        background: linear-gradient(90deg, #e94560 0%, #533483 100%);
        padding: 9px 14px;
        font-size: 0.88rem;
        font-weight: 700;
        color: #fff;
        letter-spacing: 0.4px;
        flex-shrink: 0;
        height: 36px;
        display: flex;
        align-items: center;
    }
    .grp-body {
        padding: 0;
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;  /* 靠上對齊 */
        overflow: hidden;
    }
    .grp-row {
        display: flex;
        align-items: center;
        padding: 7px 14px;
        border-bottom: 1px solid rgba(255,255,255,0.055);
        min-height: 42px;
    }
    .grp-row:last-child { border-bottom: none; }
    .grp-flag { width: 28px; height: 21px; margin-right: 8px; flex-shrink: 0; object-fit: cover; border-radius: 2px; vertical-align: middle; }
    .grp-nm {
        font-size: 0.82rem;
        font-weight: 600;
        color: #e8e8e8;
        line-height: 1.3;
        word-break: break-word;
        overflow-wrap: anywhere;
        flex: 1;
    }
    .grp-en {
        font-size: 0.70rem;
        color: #7a9ab5;
        line-height: 1.25;
        margin-left: 4px;
        flex-shrink: 0;
        max-width: 90px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .grp-rank {
        font-size: 0.68rem;
        color: #c08aff;
        background: rgba(138,43,226,0.18);
        border: 1px solid rgba(138,43,226,0.28);
        padding: 1px 7px;
        border-radius: 20px;
        flex-shrink: 0;
        margin-left: 4px;
        white-space: nowrap;
    }
    </style>
    """
    st.markdown(GRID_CSS, unsafe_allow_html=True)

    st.markdown("---")

    # 建 Grid HTML
    cards_html = ""
    for g, teams in WC_2026_GROUPS.items():
        rows_html = ""
        for t in teams:
            info = TEAM_INFO.get(t, {'flag': '🏳️', 'iso': 'un', 'cn': t, 'en': t, 'fifa_rank': 99})
            iso = info.get('iso', 'un')
            flag_url = f"https://flagcdn.com/28x21/{iso}.png"
            rows_html += f"""<div class="grp-row">
                <img class="grp-flag" src="{flag_url}" alt="{info['en']}" onerror="this.style.display='none'">
                <span class="grp-nm">{info['cn']}</span>
                <span class="grp-en">({info['en']})</span>
                <span class="grp-rank">#{info['fifa_rank']}</span>
            </div>"""
        cards_html += f"""<div class="grp-card">
            <div class="grp-hdr">🏟️ Group {g}</div>
            <div class="grp-body">{rows_html}</div>
        </div>"""

    st.markdown(f'<div class="grp-grid">{cards_html}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔬 模型架構")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🤖 **分類模型**\n\nXGBoost — 勝/平/負\n\n• 時間衰減權重\n• FIFA排名差異\n• 球隊狀態指標")
    with col2:
        st.info("📊 **進球預測**\n\nPoisson 迴歸\n\n• 平均進球率\n• 對手防守能力\n• 歷史進球分佈")
    with col3:
        st.info("🎲 **Monte Carlo**\n\n10,000次奪冠模擬\n\n• 小組賽積分制\n• 淘汰賽隨機模擬\n• 排名加權概率")

# ============================================================
# PAGE 2: 2026 預測
# ============================================================
elif page == "🔮 2026 預測":
    st.title("🔮 2026 世界盃比分預測")
    st.markdown("**XGBoost + Poisson 迴歸 · Walk-Forward 驗證**")
    st.markdown("---")

    match_df = load_match_data()
    fifa_df  = load_fifa_ranking()

    # v2.2：先試讀預訓練 pkl，0.5 秒搞定；否則 fallback 即時訓練
    pre = load_pretrained()
    if pre is not None:
        st.caption("⚡ 使用預訓練模型（models/*.pkl）— 跳過冷啟動")
        clf = pre['clf']
        poisson1 = pre['poisson1']
        poisson2 = pre['poisson2']
        feat_cols = pre['feat_cols']
        val_accs = pre['val_accs']
        avg_val_acc = np.mean([v for v in val_accs.values() if v is not None]) if val_accs else 0.5
        test_acc = val_accs.get(2022) if val_accs else None
        result = (clf, avg_val_acc, test_acc, poisson1, poisson2, feat_cols, val_accs)
    else:
        result = train_models_walkforward(match_df, fifa_df)

    if result[0] is None:
        st.warning("⚠️ 訓練資料不足，請確認網路連線後重試")
    else:
        clf, avg_val_acc, test_acc, poisson1, poisson2, feat_cols, val_accs = result

        # Walk-Forward 準確率表格
        st.subheader("📊 Walk-Forward 驗證結果")
        wf_cols = st.columns(4)
        years = [2010, 2014, 2018, 2022]
        for i, yr in enumerate(years):
            acc = val_accs.get(yr)
            with wf_cols[i]:
                if acc is not None:
                    st.metric(f"{yr} WC", f"{acc:.1%}")
                else:
                    st.metric(f"{yr} WC", "N/A")

        st.markdown("---")
        col_acc, col_base = st.columns(2)
        with col_acc:
            st.metric("平均驗證準確率", f"{avg_val_acc:.1%}", delta=f"+{avg_val_acc-0.333:.1%} vs 隨機基準")
        with col_base:
            st.metric("隨機猜測基準", "33.3%", help="三分類（勝/平/負）隨機猜測的理論準確率")
        st.caption("以歷史國際賽（1990-含預測年前）訓練，預測該屆世界盃小組賽 · 三分類隨機基準 = 33.3%")

        selected_group = st.selectbox("🏟️ 選擇小組", list(WC_2026_GROUPS.keys()))
        teams = WC_2026_GROUPS[selected_group]

        st.markdown(f"### 🏟️ Group {selected_group} 比賽預測")
        st.markdown("---")

        results = []
        for i, t1 in enumerate(teams):
            for t2 in teams[i+1:]:
                try:
                    pred = predict_match(t1, t2, 2026, match_df, fifa_df, clf, poisson1, poisson2, feat_cols)
                    results.append(pred)
                except Exception as e:
                    continue

        results.sort(key=lambda x: x['win_prob'], reverse=True)

        for r in results:
            # 用分類器機率決定標籤（主要依據），Poisson 進球只當比分參考
            info1 = TEAM_INFO.get(r['team1'], {'iso': 'un', 'cn': r['team1']})
            info2 = TEAM_INFO.get(r['team2'], {'iso': 'un', 'cn': r['team2']})
            iso1 = info1.get('iso', 'un')
            iso2 = info2.get('iso', 'un')
            flag1_html = f'<img src="https://flagcdn.com/40x30/{iso1}.png" style="height:36px;border-radius:3px;vertical-align:middle;">'
            flag2_html = f'<img src="https://flagcdn.com/40x30/{iso2}.png" style="height:36px;border-radius:3px;vertical-align:middle;">'

            # 以機率最大值決定結果標籤
            probs = {'win': r['win_prob'], 'draw': r['draw_prob'], 'loss': r['loss_prob']}
            outcome = max(probs, key=probs.get)
            if outcome == 'win':
                label_html = f"🏆 <b>{info1['cn']} 勝</b>"
                label_color = "#f7c948"
            elif outcome == 'loss':
                label_html = f"🏆 <b>{info2['cn']} 勝</b>"
                label_color = "#f7c948"
            else:
                label_html = "⚖️ <b>和局</b>"
                label_color = "#aabbcc"

            st.markdown(f"""
            <div class="pred-card">
              <div style="display:flex; align-items:center; gap:14px; margin-bottom:10px;">
                {flag1_html}
                <span style="font-size:1.1rem; color:#8899aa; font-weight:600;">VS</span>
                {flag2_html}
              </div>
              <div style="font-size:1.05rem; font-weight:600; color:{label_color}; margin-bottom:6px;">
                {label_html}
              </div>
              <div style="color:#00d4ff; font-size:1.2rem; font-weight:700;">
                比分預測：{r['goal1']} - {r['goal2']}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Progress bars（三欄：team1勝 / 和局 / team2勝）
            c1p, c2p, c3p = st.columns(3)
            with c1p:
                st.progress(r['win_prob'], text=f"{info1['cn']} 勝 {r['win_prob']:.0%}")
            with c2p:
                st.progress(r['draw_prob'], text=f"和局 {r['draw_prob']:.0%}")
            with c3p:
                st.progress(r['loss_prob'], text=f"{info2['cn']} 勝 {r['loss_prob']:.0%}")
            st.markdown("---")

# ============================================================
# PAGE 3: 數據分析
# ============================================================
elif page == "📈 數據分析":
    st.title("📈 歷史數據分析")
    st.markdown("**49,328 場真實國際比賽（1872-2026）**")
    st.markdown("---")

    match_df = load_match_data()
    wc = match_df[match_df['tournament'].str.contains('World Cup', na=False)]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("⚽ 世界盃進球分佈")
        all_g = pd.concat([wc['home_score'], wc['away_score']])
        fig = px.histogram(all_g, nbins=9, title="進球數分佈",
                          labels={'value': '進球數', 'count': '場次'},
                          color_discrete_sequence=['#3366cc'])
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("📅 每場平均進球趨勢")
        yearly = match_df[match_df['year'] >= 1970].groupby('year').agg(
            total=('home_score', lambda x: x.sum() + match_df.loc[x.index, 'away_score'].sum()),
            count=('home_score', 'count')
        )
        yearly['avg'] = yearly['total'] / yearly['count']
        yearly = yearly.reset_index()
        fig = px.line(yearly, x='year', y='avg', title='每場平均總進球（1970年至今）',
                     labels={'year': '年份', 'avg': '平均進球'})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🏆 各屆世界盃參賽球隊實力（2026分組）")

    # 2026球隊的歷史表現
    team_hist = []
    for g, teams in WC_2026_GROUPS.items():
        for t in teams:
            s = compute_team_strength(match_df, t, 2026)
            info = TEAM_INFO.get(t, {'flag': '🏳️', 'cn': t})
            team_hist.append({
                'flag_cn': f"{info['flag']} {info['cn']}",
                'Team': t,
                'Group': g,
                'Win Rate': f"{s['win_rate']:.1%}",
                'Avg Goals': f"{s['avg_goals']:.2f}",
                'Matches': s['matches']
            })

    hist_df = pd.DataFrame(team_hist).sort_values(['Group', 'Win Rate'], ascending=[True, False])
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

    # ── 模型評估區塊（v2.3 新增）──
    st.markdown("---")
    st.subheader("🎯 XGBoost 模型評估（2022 WC 測試集）")
    em = load_eval_metrics()
    if em is None:
        st.info("⚠️ 找不到 models/eval_metrics.pkl，請先在本機跑 `python pretrain.py` 產生評估數據。")
    else:
        _acc = em['accuracy']
        _n = em['n_test']
        st.markdown(
            f"測試集：**2022 世界盃小組賽** · 共 **{_n}** 場 · "
            f"整體準確率 **{_acc:.1%}**（三向分類：主隊勝 / 平 / 主隊負）"
        )
        tab_cm, tab_roc, tab_cal, tab_fi = st.tabs(
            ["📊 Confusion Matrix", "📈 ROC 曲線", "🎚 Calibration", "🔍 特徵重要性"]
        )

        # ── Tab 1: Confusion Matrix ──
        with tab_cm:
            cm_data = em['cm']
            lbls = em['labels_name']
            import plotly.graph_objects as _go
            # 標準化為百分比（row-wise）
            cm_norm = [[round(v / max(sum(row), 1) * 100, 1) for v in row] for row in cm_data]
            text_vals = [
                [f"{cm_data[i][j]}<br>({cm_norm[i][j]}%)" for j in range(3)]
                for i in range(3)
            ]
            fig_cm = _go.Figure(data=_go.Heatmap(
                z=[[cm_data[i][j] for j in range(3)] for i in range(3)],
                x=lbls, y=lbls,
                colorscale='Reds',
                text=text_vals,
                texttemplate="%{text}",
                textfont={"size": 14},
                showscale=True,
            ))
            fig_cm.update_layout(
                title=f"Confusion Matrix — 2022 WC (acc={_acc:.2f})",
                xaxis_title="Predicted", yaxis_title="Actual",
                height=420,
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            st.caption("格子內：場數（列百分比）。對角線為預測正確；主要誤差在平局辨識。")

        # ── Tab 2: ROC 曲線 ──
        with tab_roc:
            import plotly.graph_objects as _go2
            fig_roc = _go2.Figure()
            colors_roc = ['#e94560', '#0f6e6e', '#3366cc']
            for idx, (name, rd) in enumerate(em['roc'].items()):
                fig_roc.add_trace(_go2.Scatter(
                    x=rd['fpr'], y=rd['tpr'],
                    mode='lines',
                    name=f"{name} (AUC = {rd['auc']:.2f})",
                    line=dict(width=2.5, color=colors_roc[idx % 3]),
                ))
            fig_roc.add_trace(_go2.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                line=dict(dash='dash', color='gray', width=1),
                showlegend=False,
            ))
            fig_roc.update_layout(
                title="ROC Curves — One-vs-Rest（2022 WC 測試集）",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=420,
                legend=dict(x=0.62, y=0.08),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.caption("AUC > 0.7 表示模型對各類別有明顯鑑別力；平局（Draw）AUC 最低，符合足球平局難預測的直覺。")

        # ── Tab 3: Calibration ──
        with tab_cal:
            import plotly.graph_objects as _go3
            cal = em['calibration']
            fig_cal = _go3.Figure()
            fig_cal.add_trace(_go3.Scatter(
                x=cal['prob_pred'], y=cal['prob_true'],
                mode='lines+markers',
                name='XGBoost（勝）',
                line=dict(color='#e94560', width=2.5),
                marker=dict(size=8),
            ))
            fig_cal.add_trace(_go3.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                line=dict(dash='dash', color='gray', width=1),
                name='完美校準',
            ))
            fig_cal.update_layout(
                title="Calibration Curve — 主隊勝（Win）類別",
                xaxis_title="預測機率",
                yaxis_title="實際頻率",
                height=420,
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_cal, use_container_width=True)
            st.caption("點越靠近對角虛線，機率預測越可靠。偏上方表示模型預測偏保守（低估勝率），偏下方則過度自信。")

        # ── Tab 4: Feature Importance ──
        with tab_fi:
            import plotly.graph_objects as _go4
            fi = em['feature_importance']
            fig_fi = _go4.Figure(_go4.Bar(
                x=fi['values'], y=fi['features'],
                orientation='h',
                marker_color='#3366cc',
            ))
            fig_fi.update_layout(
                title="Feature Importance（XGBoost gain）",
                xaxis_title="Importance",
                height=max(350, len(fi['features']) * 28),
                margin=dict(l=160),
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            st.caption("數值越高代表該特徵對模型決策影響越大。FIFA 排名差距（rank_diff）通常是最強預測因子。")

# ============================================================
# PAGE 4: 球隊風格分群（v2.2 新增 — 滿足課程「分群」任務）
# ============================================================
elif page == "🎯 球隊風格分群":
    st.title("🎯 球隊風格分群")
    st.markdown("**K-Means + PCA 二維投影 · 對 48 支 2026 參賽國的攻守風格做非監督式分群**")
    st.markdown("---")

    clusters = load_team_clusters()
    if clusters is None:
        st.warning("⚠️ 找不到 models/team_clusters.pkl，請先在本機跑 `python pretrain.py`。")
    else:
        df_c = clusters['df']
        cluster_names = clusters['cluster_names']
        centers = clusters['centers']
        sil = clusters['silhouette']
        k = clusters['k']

        # ── 概覽指標 ──
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><h2>{k}</h2><p>分群數 (k)</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><h2>{sil:.2f}</h2><p>Silhouette 分數</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><h2>48</h2><p>球隊樣本</p></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><h2>7</h2><p>輸入特徵</p></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🗺️ PCA 2D 投影（每個點 = 一支球隊）")

        # ── Plotly PCA 散點 ──
        plot_df = df_c.reset_index()
        plot_df['cluster_label'] = plot_df['cluster'].map(cluster_names)
        plot_df['team_cn'] = plot_df['team'].apply(lambda t: TEAM_INFO.get(t, {'cn': t})['cn'])
        plot_df['flag'] = plot_df['team'].apply(lambda t: TEAM_INFO.get(t, {'flag': '🏳️'})['flag'])
        plot_df['hover_name'] = plot_df['flag'] + ' ' + plot_df['team_cn']

        fig_pca = px.scatter(
            plot_df, x='PC1', y='PC2',
            color='cluster_label',
            hover_name='hover_name',
            text='flag',
            title=f'球隊風格 PCA 分群（k={k}，Silhouette={sil:.2f}）',
            color_discrete_sequence=['#e94560', '#0f6e6e', '#3366cc', '#f5a623'],
            height=520,
        )
        fig_pca.update_traces(textposition='top center', marker=dict(size=12))
        fig_pca.update_layout(legend_title_text='風格群')
        st.plotly_chart(fig_pca, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📡 分群靜態雷達圖")
        radar_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', '14_cluster_radar.png')
        pca_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', '15_cluster_pca.png')
        col_r, col_p = st.columns(2)
        with col_r:
            if os.path.exists(radar_path):
                st.image(radar_path, caption='各群風格雷達圖（攻守特徵）', use_column_width=True)
        with col_p:
            if os.path.exists(pca_path):
                st.image(pca_path, caption='PCA 靜態散點（含球隊標籤）', use_column_width=True)

        st.markdown("---")
        st.markdown("### 📋 各群球隊列表")
        for cid, cname in cluster_names.items():
            teams_in_c = plot_df[plot_df['cluster'] == cid]['team'].tolist()
            team_labels = []
            for t in sorted(teams_in_c):
                info = TEAM_INFO.get(t, {'flag': '🏳️', 'cn': t})
                team_labels.append(f"{info['flag']} {info['cn']}")
            with st.expander(f"**{cname}**（{len(teams_in_c)} 支球隊）"):
                st.write(' · '.join(team_labels))

        st.markdown("---")
        st.markdown("### 🔍 兩隊風格比較")
        all_teams_sorted = sorted(plot_df['team'].tolist())
        team_options = [f"{TEAM_INFO.get(t,{'flag':'🏳️'})['flag']} {TEAM_INFO.get(t,{'cn':t})['cn']} ({t})" for t in all_teams_sorted]
        col_a, col_b = st.columns(2)
        with col_a:
            sel_a = st.selectbox("選擇球隊 A", team_options, index=0, key='cluster_team_a')
        with col_b:
            sel_b = st.selectbox("選擇球隊 B", team_options, index=1, key='cluster_team_b')

        team_a = sel_a.split('(')[-1].rstrip(')')
        team_b = sel_b.split('(')[-1].rstrip(')')

        feat_cols_radar = [c for c in df_c.columns if c not in ('cluster', 'PC1', 'PC2')]
        if feat_cols_radar and team_a in df_c.index and team_b in df_c.index:
            row_a = df_c.loc[team_a, feat_cols_radar]
            row_b = df_c.loc[team_b, feat_cols_radar]
            # 正規化到 0-1
            col_min = df_c[feat_cols_radar].min()
            col_max = df_c[feat_cols_radar].max()
            norm_a = ((row_a - col_min) / (col_max - col_min + 1e-9)).tolist()
            norm_b = ((row_b - col_min) / (col_max - col_min + 1e-9)).tolist()
            import plotly.graph_objects as _go_r
            fig_radar = _go_r.Figure()
            fig_radar.add_trace(_go_r.Scatterpolar(
                r=norm_a + [norm_a[0]], theta=feat_cols_radar + [feat_cols_radar[0]],
                fill='toself', name=TEAM_INFO.get(team_a, {'cn': team_a})['cn'],
                line_color='#e94560',
            ))
            fig_radar.add_trace(_go_r.Scatterpolar(
                r=norm_b + [norm_b[0]], theta=feat_cols_radar + [feat_cols_radar[0]],
                fill='toself', name=TEAM_INFO.get(team_b, {'cn': team_b})['cn'],
                line_color='#3366cc', opacity=0.7,
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True, height=450,
                title='兩隊攻守風格雷達比較（正規化）',
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# ============================================================
# PAGE 5: 奪冠預測（Monte Carlo）
# ============================================================
elif page == "🏅 奪冠預測":
    st.title("🏅 Monte Carlo 奪冠機率預測")
    st.markdown("**模擬 2026 世界盃整個賽程，含小組賽 → 32 強 → 16 強 → 8 強 → 4 強 → 決賽**")
    st.markdown("---")

    mc = load_mc_results()
    if mc is None:
        st.warning("⚠️ 找不到 models/mc_results.pkl，請先在本機跑 `python pretrain.py`。")
    else:
        n_sims = mc.get('n_sims', 5000)
        results = mc.get('results', mc)  # 相容兩種格式

        rows = []
        for team, d in results.items():
            if team == 'n_sims':
                continue
            info = TEAM_INFO.get(team, {'flag': '🏳️', 'cn': team})
            rows.append({
                'flag_cn': f"{info['flag']} {info['cn']}",
                'team': team,
                'champ_pct': d.get('champ_pct', 0),
                'final_pct': d.get('final_pct', 0),
                'semi_pct': d.get('semi_pct', 0),
            })

        mc_df = pd.DataFrame(rows).sort_values('champ_pct', ascending=False).reset_index(drop=True)
        mc_df.index += 1

        st.markdown(f"*模擬次數：{n_sims:,} 次*")
        c1, c2, c3 = st.columns(3)
        if len(mc_df) > 0:
            top = mc_df.iloc[0]
            with c1:
                st.markdown(f'<div class="metric-card"><h2>{top["flag_cn"]}</h2><p>奪冠熱門 {top["champ_pct"]:.1%}</p></div>', unsafe_allow_html=True)
            if len(mc_df) > 1:
                with c2:
                    t2 = mc_df.iloc[1]
                    st.markdown(f'<div class="metric-card"><h2>{t2["flag_cn"]}</h2><p>第二熱門 {t2["champ_pct"]:.1%}</p></div>', unsafe_allow_html=True)
            if len(mc_df) > 2:
                with c3:
                    t3 = mc_df.iloc[2]
                    st.markdown(f'<div class="metric-card"><h2>{t3["flag_cn"]}</h2><p>第三熱門 {t3["champ_pct"]:.1%}</p></div>', unsafe_allow_html=True)

        st.markdown("---")
        top20 = mc_df.head(20)
        fig_mc = px.bar(
            top20, x='champ_pct', y='flag_cn',
            orientation='h',
            title=f'奪冠機率 Top 20（Monte Carlo {n_sims:,} 次模擬）',
            labels={'champ_pct': '奪冠機率', 'flag_cn': '球隊'},
            color='champ_pct',
            color_continuous_scale='Reds',
            text=top20['champ_pct'].apply(lambda x: f'{x:.1%}'),
        )
        fig_mc.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False,
            height=600,
        )
        fig_mc.update_traces(textposition='outside')
        st.plotly_chart(fig_mc, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 各階段晉級機率")
        display_df = mc_df[['flag_cn', 'champ_pct', 'final_pct', 'semi_pct']].copy()
        display_df.columns = ['球隊', '奪冠', '進決賽', '進四強']
        for col in ['奪冠', '進決賽', '進四強']:
            display_df[col] = display_df[col].apply(lambda x: f'{x:.1%}')
        st.dataframe(display_df, use_container_width=True)

# ============================================================
# PAGE 6: 完整賽程
# ============================================================
elif page == "📅 完整賽程":
    st.title("📅 2026 世界盃完整賽程")
    st.markdown("**含 12 個小組分組 · 48 支球隊 · 小組賽 + 淘汰賽路徑**")
    st.markdown("---")

    pre = load_pretrained()

    st.markdown("## 🏟️ 小組賽分組")
    for group, teams in WC_2026_GROUPS.items():
        with st.expander(f"**第 {group} 組**（{' vs '.join([TEAM_INFO.get(t,{'cn':t})['cn'] for t in teams])}）"):
            match_df_g = load_match_data()
            group_rows = []
            for i, t1 in enumerate(teams):
                for t2 in teams[i+1:]:
                    info1 = TEAM_INFO.get(t1, {'flag': '🏳️', 'cn': t1})
                    info2 = TEAM_INFO.get(t2, {'flag': '🏳️', 'cn': t2})
                    if pre:
                        try:
                            feat = create_features(t1, t2, 2026, match_df_g, load_fifa_ranking(),
                                                   pre['feat_cols'])
                            X = pd.DataFrame([feat])[pre['feat_cols']]
                            prob = pre['clf'].predict_proba(X)[0]
                            lam1 = max(0.1, float(pre['poisson1'].predict(X)[0]))
                            lam2 = max(0.1, float(pre['poisson2'].predict(X)[0]))
                            score1 = round(lam1, 1)
                            score2 = round(lam2, 1)
                            result_str = f"{score1:.1f} - {score2:.1f}"
                        except Exception:
                            prob = [0.33, 0.33, 0.34]
                            result_str = "N/A"
                    else:
                        prob = [0.33, 0.33, 0.34]
                        result_str = "需先跑 pretrain.py"
                    group_rows.append({
                        '對戰': f"{info1['flag']} {info1['cn']} vs {info2['flag']} {info2['cn']}",
                        '預測比分': result_str,
                        '主勝%': f"{prob[2]:.0%}",
                        '平%': f"{prob[1]:.0%}",
                        '主負%': f"{prob[0]:.0%}",
                    })
            if group_rows:
                st.dataframe(pd.DataFrame(group_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("## 🏆 淘汰賽路徑說明")
    st.markdown("""
    2026 世界盃採 48 隊制：
    - **小組賽**：12 組，每組 4 隊，取前 2 名 + 8 支最佳第三名 → 共 32 強
    - **32 強淘汰賽** → **16 強** → **8 強（四分之一決賽）** → **四強（半決賽）** → **決賽**
    - 奪冠機率請參考「🏅 奪冠預測」頁的 Monte Carlo 結果
    """)

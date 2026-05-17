"""
🏆 2026 世界盃 ML 勝率分析與比分預測系統 v2.1
World Cup 2026 — ML Win Probability & Score Prediction
============================================================
真實資料：49,328場國際賽事 + 67,894筆FIFA排名
"""

import os
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
from squad_data import SQUAD_DATA

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

/* ── 版面優化：統一字級／間距／視覺層級 ── */

/* 主容器內距 */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* 標題層級 */
h1 {
    font-weight: 800 !important;
    letter-spacing: -0.5px;
    border-bottom: 3px solid var(--accent-red);
    padding-bottom: 10px;
    margin-bottom: 18px !important;
}
h2 {
    font-weight: 700 !important;
    color: #f7c948 !important;
    margin-top: 1.5rem !important;
}
h3 {
    font-weight: 700 !important;
    color: var(--accent-cyan) !important;
    margin-top: 1.2rem !important;
}

/* 分隔線更精緻 */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(233,69,96,0.4), transparent) !important;
    margin: 1.6rem 0 !important;
}

/* Tabs 樣式：更明顯的選中狀態 */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(18,18,26,0.6);
    padding: 6px;
    border-radius: 10px;
    border: 1px solid rgba(233,69,96,0.1);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
    color: #8899aa;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(233,69,96,0.08);
    color: #f0f0f0;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #e94560 0%, #c23a52 100%) !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(233,69,96,0.3);
}

/* Metric 卡片更立體 */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #12121a 0%, #1a1a2e 100%);
    border: 1px solid rgba(233,69,96,0.15);
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}
div[data-testid="stMetricLabel"] {
    color: #8899aa !important;
    font-weight: 600;
}
div[data-testid="stMetricValue"] {
    color: var(--accent-cyan) !important;
    font-weight: 800 !important;
}

/* Dataframe 樣式 */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(233,69,96,0.12);
}

/* Expander：強化視覺辨識（原本太透明看不見） */
div[data-testid="stExpander"] {
    border: 1px solid rgba(0,212,255,0.3) !important;
    border-radius: 10px !important;
    background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    margin-bottom: 8px;
}
div[data-testid="stExpander"] summary {
    font-weight: 700 !important;
    color: #ffffff !important;
    padding: 12px 18px !important;
    background: linear-gradient(90deg, rgba(0,212,255,0.12) 0%, transparent 100%) !important;
    border-left: 3px solid #00d4ff !important;
}
div[data-testid="stExpander"] summary:hover {
    background: linear-gradient(90deg, rgba(0,212,255,0.22) 0%, transparent 100%) !important;
}

/* Info / Warning / Error 訊息框：強化對比 */
div[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-weight: 600;
    border-width: 1px !important;
    border-style: solid !important;
}
div[data-testid="stAlert"][data-baseweb="notification"] {
    background: rgba(0,212,255,0.12) !important;
    border-color: rgba(0,212,255,0.4) !important;
    color: #ffffff !important;
}

/* ── 球隊對比表 cell classes（避開 Streamlit inline-style 被剝離問題） ── */
.cmp-grid {
    display: grid;
    grid-template-columns: 1.4fr 1fr 1fr;
    margin-top: 10px;
    background: #ffffff;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
.cmp-head {
    background: #f1f5f9 !important;
    color: #0f172a !important;
    font-weight: 700;
    padding: 12px 14px;
    border-bottom: 2px solid #cbd5e1;
}
.cmp-head-c { text-align: center; }
.cmp-head-red { border-bottom: 3px solid #dc2626 !important; }
.cmp-head-blue { border-bottom: 3px solid #2563eb !important; }
.cmp-label {
    background: #ffffff !important;
    color: #0f172a !important;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 14px;
    border-top: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
}
.cmp-cell {
    background: #ffffff !important;
    color: #0f172a !important;
    font-weight: 700;
    text-align: center;
    padding: 14px;
    font-size: 1.05rem;
    border-top: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
}
.cmp-dim {
    background: #ffffff !important;
    color: #94a3b8 !important;
    font-weight: 600;
    text-align: center;
    padding: 14px;
    font-size: 1.05rem;
    border-top: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
}
.cmp-win {
    background: #fde047 !important;
    color: #000000 !important;
    font-weight: 900 !important;
    text-align: center;
    padding: 14px;
    font-size: 1.15rem;
    border-top: 3px solid #ea580c !important;
    border-bottom: 3px solid #ea580c !important;
}
.cmp-dir {
    color: #ea580c;
    font-weight: 700;
    margin-right: 6px;
}
.cmp-chip {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    margin-right: 6px;
    vertical-align: middle;
}
.cmp-chip-red { background: #dc2626; }
.cmp-chip-blue { background: #2563eb; }

/* Sidebar 樣式 */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a0f 0%, #12121a 100%);
    border-right: 1px solid rgba(233,69,96,0.18);
}
section[data-testid="stSidebar"] .stRadio label {
    padding: 6px 4px;
    border-radius: 8px;
    transition: background 0.15s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(233,69,96,0.08);
}

/* Selectbox / Multiselect */
div[data-baseweb="select"] > div {
    background: rgba(26,26,46,0.8) !important;
    border-color: rgba(233,69,96,0.2) !important;
}

/* Button */
.stButton button {
    background: linear-gradient(135deg, #e94560 0%, #c23a52 100%);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 18px;
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(233,69,96,0.4);
}

/* Caption 樣式 */
[data-testid="stCaptionContainer"] {
    color: #7a8aa0 !important;
    font-size: 0.82rem !important;
    line-height: 1.6;
}

/* Plotly 圖表外框 */
.js-plotly-plot {
    border-radius: 10px;
    background: rgba(18,18,26,0.4);
    padding: 8px;
}

/* Progress bar */
div[data-testid="stProgress"] > div > div {
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
}

/* 響應式：手機版減少 padding */
@media (max-width: 768px) {
    .main .block-container { padding: 1rem 0.6rem; }
    h1 { font-size: 1.6rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.1rem !important; }
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
    'Morocco':        {'flag': '🇲🇦', 'iso': 'ma', 'cn': '摩洛哥',         'en': 'Morocco',         'fifa_rank': 14, 'fifa_pts': 1720},
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

_SQUAD_OVR_BASELINE = 79.0  # 世界盃平均球隊主將 OVR 基準

def squad_ovr(team: str) -> float:
    """回傳球隊主將平均 OVR；若無資料則回傳基準值"""
    players = SQUAD_DATA.get(team, [])
    if not players:
        return _SQUAD_OVR_BASELINE
    return float(np.mean([p['ovr'] for p in players]))

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

# 小組賽賽程（台灣時間 UTC+8）
# 各組隊伍索引對應 WC_2026_GROUPS 中的順序 0-3
# 每組三輪：MD1 (0v1, 2v3)、MD2 (0v2, 1v3)、MD3 (0v3, 1v2，最後一輪同時開踢)
def _build_group_schedule() -> dict:
    md1 = {'A':'6/12','B':'6/13','C':'6/14','D':'6/14',
            'E':'6/15','F':'6/15','G':'6/16','H':'6/16',
            'I':'6/17','J':'6/17','K':'6/18','L':'6/18'}
    md2 = {'A':'6/21','B':'6/22','C':'6/22','D':'6/23',
            'E':'6/23','F':'6/24','G':'6/24','H':'6/25',
            'I':'6/25','J':'6/26','K':'6/26','L':'6/27'}
    md3 = {'A':'6/29','B':'6/29','C':'6/30','D':'6/30',
            'E':'7/1', 'F':'7/1', 'G':'7/1', 'H':'7/2',
            'I':'7/2', 'J':'7/2', 'K':'7/2', 'L':'7/2'}
    s = {}
    for g in 'ABCDEFGHIJKL':
        s[(g,0,1)] = f"{md1[g]} 09:00"
        s[(g,2,3)] = f"{md1[g]} 22:00"
        s[(g,0,2)] = f"{md2[g]} 09:00"
        s[(g,1,3)] = f"{md2[g]} 22:00"
        s[(g,0,3)] = f"{md3[g]} 04:00"
        s[(g,1,2)] = f"{md3[g]} 04:00"
    return s

_GROUP_SCHEDULE = _build_group_schedule()

# ============================================================
# DATA LOADING
# ============================================================
# 資料集中的官方名稱 → 本 app 使用的名稱（統一鍵值）
_DATASET_NAME_MAP = {
    'Korea Republic':          'South Korea',
    'Côte d\'Ivoire':          'Ivory Coast',
    "Cote d'Ivoire":           'Ivory Coast',
    'Türkiye':                 'Turkiye',
    'Turkey':                  'Turkiye',
    'Czech Republic':          'Czechia',
    'Congo DR':                'DR Congo',
    'Democratic Republic of the Congo': 'DR Congo',
    'United States':           'USA',
    'Bosnia-Herzegovina':      'Bosnia and Herzegovina',
}

def _norm_teams(df, cols):
    for c in cols:
        df[c] = df[c].replace(_DATASET_NAME_MAP)
    return df

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
    df = _norm_teams(df, ['home_team', 'away_team'])
    return df

@st.cache_data
def load_fifa_ranking():
    """載入 FIFA 排名歷史（直接從URL讀取）"""
    url = "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/ranking_fifa_historical.csv"
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df['date'])
    df = _norm_teams(df, ['team'])
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

    is_home = team_matches['home_team'] == team
    gf = np.where(is_home, team_matches['home_score'], team_matches['away_score'])
    ga = np.where(is_home, team_matches['away_score'], team_matches['home_score'])
    weight = np.exp(-0.1 * (year - team_matches['year']))
    total_weight = weight.sum()

    win_rate = weight[gf > ga].sum() / total_weight
    draw_rate = weight[gf == ga].sum() / total_weight
    avg_goals = (gf * weight).sum() / total_weight
    avg_conceded = (ga * weight).sum() / total_weight

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
    is_home = recent['home_team'] == team
    wins = (
        (is_home & (recent['home_score'] > recent['away_score'])) |
        (~is_home & (recent['away_score'] > recent['home_score']))
    ).sum()
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
    'DR Congo': 'CAF', 'South Africa': 'CAF', 'Cameroon': 'CAF', 'Nigeria': 'CAF',
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
                except Exception:
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
        except Exception:
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
                except Exception:
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

@st.cache_resource(ttl=600)
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
    """預測單場（中性場地對稱修正 + Dixon-Coles 期望進球）"""

    def _probs(t1, t2):
        feat = create_features_v2(t1, t2, year, match_df, fifa_df)
        X = pd.DataFrame([feat])[feat_cols]
        proba = clf.predict_proba(X)[0]
        pw = pd_ = pl = 1/3
        for c, p in zip(clf.classes_, proba):
            if c == 2: pw = p
            elif c == 1: pd_ = p
            else: pl = p
        t = pw + pd_ + pl
        return pw/t, pd_/t, pl/t

    # ── 第一層：XGBoost（正向+反向對稱平均，消除主場偏差）──
    pw_f, pd_f, pl_f = _probs(team1, team2)
    pw_r, pd_r, pl_r = _probs(team2, team1)
    clf_win  = (pw_f + pl_r) / 2
    clf_draw = (pd_f + pd_r) / 2
    clf_loss = (pl_f + pw_r) / 2
    _t = clf_win + clf_draw + clf_loss
    clf_win /= _t; clf_draw /= _t; clf_loss /= _t

    # ── 第二層：Dixon-Coles 期望進球（足聯品質校正 + FIFA積分調整）──
    s1 = compute_team_strength(match_df, team1, year)
    s2 = compute_team_strength(match_df, team2, year)
    LEAGUE_AVG = 1.35
    CONFED_SCALE = {'UEFA': 0.96, 'CONMEBOL': 1.00, 'CONCACAF': 0.88,
                    'AFC': 0.84, 'CAF': 0.78, 'OFC': 0.72}
    cs1 = CONFED_SCALE.get(CONFED_MAP.get(team1, 'CAF'), 0.84)
    cs2 = CONFED_SCALE.get(CONFED_MAP.get(team2, 'CAF'), 0.84)

    atk1 = min(2.0, s1['avg_goals'] * cs1)
    atk2 = min(2.0, s2['avg_goals'] * cs2)
    def2 = max(0.75, s2['avg_conceded'] * (cs2 ** 0.5))
    def1 = max(0.75, s1['avg_conceded'] * (cs1 ** 0.5))
    lam1_dc = min(2.2, max(0.4, atk1 * LEAGUE_AVG / def2))
    lam2_dc = min(2.2, max(0.3, atk2 * LEAGUE_AVG / def1))

    pts1, pts2 = team_pts(team1), team_pts(team2)
    rank_factor = ((pts1 + 300) / (pts2 + 300)) ** 0.20
    lam1 = min(2.2, max(0.4, lam1_dc * rank_factor))
    lam2 = min(2.2, max(0.3, lam2_dc / rank_factor))

    # ── 主將 OVR 調整（第四層）：依主將平均能力值輕微修正 λ ──
    # 幂次 0.18 保守調整，避免單一球星過度主導結果
    ovr1, ovr2 = squad_ovr(team1), squad_ovr(team2)
    squad_factor = (ovr1 / ovr2) ** 0.18
    lam1 = min(2.2, max(0.4, lam1 * squad_factor))
    lam2 = min(2.2, max(0.3, lam2 / squad_factor))

    # 從 λ 積分出 Poisson 勝/平/負機率（0~7 進球範圍涵蓋 99.9%+ 機率）
    poi_win = poi_draw = poi_loss = 0.0
    for g1 in range(8):
        for g2 in range(8):
            p = poisson.pmf(g1, lam1) * poisson.pmf(g2, lam2)
            if g1 > g2:
                poi_win += p
            elif g1 == g2:
                poi_draw += p
            else:
                poi_loss += p
    _t2 = poi_win + poi_draw + poi_loss
    poi_win /= _t2; poi_draw /= _t2; poi_loss /= _t2

    # ── 第三層：加權融合（XGBoost 20% + Poisson 80%）──
    # Poisson 直接由 λ（球隊攻防強度）推導，與顯示的 λ₁/λ₂ 內部一致
    # XGBoost 權重降低，避免在強弱明顯時被分類器誤導反向
    W_CLF, W_POI = 0.20, 0.80
    prob_win  = W_CLF * clf_win  + W_POI * poi_win
    prob_draw = W_CLF * clf_draw + W_POI * poi_draw
    prob_loss = W_CLF * clf_loss + W_POI * poi_loss
    _t3 = prob_win + prob_draw + prob_loss
    prob_win /= _t3; prob_draw /= _t3; prob_loss /= _t3

    # ── MAP 比分：直接取整個 Poisson 矩陣的絕對最高機率 cell ──
    # 不再套用勝負方向約束 → 確保 ★ 標記就是矩陣裡實際最高的格子
    # 規範化到字母序，確保同場比賽比分不因呼叫順序而改變
    t_can1, t_can2 = sorted([team1, team2])
    is_canonical = (team1 == t_can1)
    lam_c1 = lam1 if is_canonical else lam2
    lam_c2 = lam2 if is_canonical else lam1

    best_prob, gc1, gc2 = -1.0, 0, 0
    for g1 in range(8):
        for g2 in range(8):
            p = poisson.pmf(g1, lam_c1) * poisson.pmf(g2, lam_c2)
            if p > best_prob:
                best_prob, gc1, gc2 = p, g1, g2

    goal1 = gc1 if is_canonical else gc2
    goal2 = gc2 if is_canonical else gc1

    # 校驗：若顯示的機率方向與比分方向不一致，以「比分方向」為準重新分配機率
    # （因為比分是用戶看到的最強指標，必須與機率一致）
    score_outcome = 'win' if goal1 > goal2 else ('draw' if goal1 == goal2 else 'loss')
    prob_max = max(prob_win, prob_draw, prob_loss)
    cur_outcome = ('win' if prob_max == prob_win
                   else 'draw' if prob_max == prob_draw else 'loss')
    if score_outcome != cur_outcome:
        # 當分歧時，套用較弱的調整把「比分方向」對應的機率拉到至少略高
        # 加 5% 到比分方向，按比例從其他兩類扣回
        adj = 0.05
        if score_outcome == 'win':
            prob_win += adj; prob_draw -= adj * (prob_draw / max(prob_draw + prob_loss, 1e-6))
            prob_loss -= adj * (prob_loss / max(prob_draw + prob_loss, 1e-6))
        elif score_outcome == 'loss':
            prob_loss += adj; prob_win -= adj * (prob_win / max(prob_win + prob_draw, 1e-6))
            prob_draw -= adj * (prob_draw / max(prob_win + prob_draw, 1e-6))
        # 重新正規化
        _t4 = prob_win + prob_draw + prob_loss
        prob_win /= _t4; prob_draw /= _t4; prob_loss /= _t4
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
        'lam1': lam1, 'lam2': lam2,
        's1': s1, 's2': s2,
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
        """小組賽單場模擬：動態p_draw，回傳 (t1積分, t2積分)"""
        bw1 = team_base_winrate(t1)
        bw2 = team_base_winrate(t2)
        rank_diff = abs(team_rank(t1) - team_rank(t2))
        p_draw = max(0.15, min(0.28, 0.28 - rank_diff * 0.005))
        p_win = bw1 / (bw1 + bw2) * (1 - p_draw)
        r = np.random.random()
        if r < p_win:
            return 3, 0   # t1勝
        elif r < p_win + p_draw:
            return 1, 1   # 平局
        else:
            return 0, 3   # t2勝

    def sim_ko_match(t1, t2):
        """淘汰賽單場模擬（使用 XGBoost 機率，無平局）"""
        try:
            feat = create_features_v2(t1, t2, 2026, match_df, fifa_df)
            X = pd.DataFrame([feat])[feat_cols]
            proba = clf.predict_proba(X)[0]
            classes = list(clf.classes_)
            pw = proba[classes.index(2)] if 2 in classes else 1/3
            pl = proba[classes.index(0)] if 0 in classes else 1/3
            # 淘汰賽無平局：按 win/(win+loss) 決定勝負
            p_t1_wins = pw / (pw + pl) if (pw + pl) > 0 else 0.5
        except Exception:
            p_t1_wins = 0.5
        return t1 if np.random.random() < p_t1_wins else t2

    for _ in range(n_sims):
        # ── 小組賽：取各組前2晉級 + 記錄各組第3名 ──
        sim_qualifiers = []
        third_place_info = []  # (pts, team) 各組第3名
        for g, teams in WC_2026_GROUPS.items():
            pts = {t: 0 for t in teams}
            for i, t1 in enumerate(teams):
                for t2 in teams[i+1:]:
                    p1, p2 = sim_group_match(t1, t2)
                    pts[t1] += p1; pts[t2] += p2
            sorted_pts = sorted(pts.items(), key=lambda x: x[1], reverse=True)
            qualifiers = [t for t, _ in sorted_pts[:2]]
            sim_qualifiers.extend(qualifiers)
            win_count[qualifiers[0]] += 1
            third_team, third_pts = sorted_pts[2]
            third_place_info.append((third_pts, third_team))

        # 2026 世界盃：24支小組前2 + 8支最佳第3名（按積分排序）= 32強
        best_thirds = sorted(third_place_info, reverse=True)[:8]
        bracket = sim_qualifiers + [t for _, t in best_thirds]
        np.random.shuffle(bracket)

        # ── 淘汰賽 R32 → R16 → QF → SF → Final ──
        current_round = bracket
        while len(current_round) > 1:
            next_round = []
            for i in range(0, len(current_round), 2):
                if i + 1 < len(current_round):
                    winner = sim_ko_match(current_round[i], current_round[i+1])
                    next_round.append(winner)
                else:
                    next_round.append(current_round[i])  # bye
            current_round = next_round

        champion = current_round[0]
        champion_count[champion] += 1

    result = {t: {'win_pct': win_count[t] / n_sims * 100,
                  'champ_pct': champion_count[t] / n_sims * 100}
              for t in all_teams}
    return result

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(
    """
    <div style='text-align:center; padding:6px 0 14px;'>
        <div style='font-size:2.4rem;'>🏆</div>
        <div style='font-size:1.05rem; font-weight:800; color:#f7c948; letter-spacing:0.5px;'>2026 World Cup</div>
        <div style='font-size:0.78rem; color:#8899aa; margin-top:2px;'>ML 勝率分析 v2.3</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown("**📍 導航**")
page = st.sidebar.radio("選擇頁面", [
    "📊 專題總覽",
    "🔮 2026 預測",
    "📈 數據分析",
    "🌍 各國分析",
    "🎯 球隊風格分群",
    "🏅 奪冠預測",
    "📅 完整賽程",
], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='font-size:0.74rem; color:#7a8aa0; line-height:1.7; padding: 6px 4px;'>
        <div><b style='color:#aabbcc;'>🤖 模型架構</b></div>
        <div>XGBoost 40% + Dixon-Coles 60%</div>
        <div>+ Squad OVR + Monte Carlo 10k</div>
        <br>
        <div><b style='color:#aabbcc;'>📊 訓練資料</b></div>
        <div>49,328 場 · 1990-2025</div>
        <div>67,894 筆 FIFA 排名</div>
        <br>
        <div style='font-size:0.7rem; color:#556677;'>
            資料來源：international football, FIFA<br>
            最後更新：2026-04
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("🤖 **第一層：XGBoost 分類**\n\n勝/平/負機率（權重 40%）\n\n• 時間衰減權重\n• FIFA 排名差異\n• 球隊近期狀態\n• 正向＋反向對稱平均")
    with col2:
        st.info("📊 **第二層：Dixon-Coles**\n\nPoisson 期望進球（權重 60%）\n\n• 足聯品質係數校正\n• FIFA 積分排名加權\n• 從 λ 積分勝/平/負機率")
    with col3:
        st.info("⭐ **第三層：主將 OVR 修正**\n\n主力球員綜合能力值調整\n\n• 各隊 5 名主將平均 OVR\n• 幂次 0.18 保守修正 λ\n• 巨星球隊自然獲得加成")
    with col4:
        st.info("🎲 **第四層：融合 + Monte Carlo**\n\n統一方向後最終預測\n\n• 勝負方向與比分完全一致\n• MAP 最高機率比分\n• 10,000 次全賽程奪冠模擬")

# ============================================================
# PAGE 2: 2026 預測
# ============================================================
elif page == "🔮 2026 預測":
    st.title("🔮 2026 世界盃比分預測")
    st.markdown("**XGBoost（40%）＋ Dixon-Coles Poisson（60%）＋ 主將 OVR 修正 · 四層融合模型 · Walk-Forward 驗證**")
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
            for j in range(i+1, len(teams)):
                t2 = teams[j]
                try:
                    pred = predict_match(t1, t2, 2026, match_df, fifa_df, clf, poisson1, poisson2, feat_cols)
                    pred['_match_time'] = _GROUP_SCHEDULE.get((selected_group, i, j), '')
                    _round_map = {(0,1):1,(2,3):1,(0,2):2,(1,3):2,(0,3):3,(1,2):3}
                    pred['_round'] = _round_map.get((i, j), 0)
                    results.append(pred)
                except Exception:
                    continue

        results.sort(key=lambda x: x.get('_round', 9))

        for r in results:
            info1 = TEAM_INFO.get(r['team1'], {'iso': 'un', 'cn': r['team1']})
            info2 = TEAM_INFO.get(r['team2'], {'iso': 'un', 'cn': r['team2']})
            iso1 = info1.get('iso', 'un')
            iso2 = info2.get('iso', 'un')

            # 勝者標色依比分決定（而非機率），確保顏色與顯示比分一致
            if r['goal1'] > r['goal2']:
                score1_color, score2_color = '#00d4ff', '#8899aa'
            elif r['goal1'] < r['goal2']:
                score1_color, score2_color = '#8899aa', '#00d4ff'
            else:
                score1_color = score2_color = '#f7c948'

            # ── 單場摘要列：旗幟 球隊名 預測分 VS 預測分 球隊名 旗幟 ｜ 平局機率 ──
            match_time = r.get('_match_time', '')
            match_round = r.get('_round', '')
            round_label = f"第{match_round}輪" if match_round else ''
            col_match, col_draw = st.columns([5, 1])
            with col_match:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;padding:6px 0;flex-wrap:wrap">'
                    f'<img src="https://flagcdn.com/40x30/{iso1}.png" style="height:26px;border-radius:3px;">'
                    f'<b style="font-size:1rem">{info1["cn"]}</b>'
                    f'<span style="font-size:1.5rem;font-weight:900;color:{score1_color};min-width:20px;text-align:right">{r["goal1"]}</span>'
                    f'<span style="color:#e94560;font-size:0.85rem;font-weight:700;margin:0 4px;'
                    f'padding:2px 8px;background:rgba(233,69,96,0.15);border-radius:4px">VS</span>'
                    f'<span style="font-size:1.5rem;font-weight:900;color:{score2_color};min-width:20px">{r["goal2"]}</span>'
                    f'<b style="font-size:1rem">{info2["cn"]}</b>'
                    f'<img src="https://flagcdn.com/40x30/{iso2}.png" style="height:26px;border-radius:3px;">'
                    f'<span style="color:#cbd5e1;font-size:0.82rem;margin-left:10px;font-weight:600">'
                    f'<span style="color:#f7c948">📅 {round_label}</span>'
                    f'{" · " if match_time else ""}<span style="color:#00d4ff">{match_time}</span>'
                    f'<span style="color:#94a3b8">（台灣時間）</span></span>'
                    f'</div>', unsafe_allow_html=True)
            with col_draw:
                st.markdown(
                    f'<div style="font-size:0.88rem;color:#e2e8f0;padding:8px 0;text-align:right;font-weight:600">'
                    f'平局機率 <b style="color:#f7c948;font-size:1.05rem">{r["draw_prob"]:.0%}</b></div>',
                    unsafe_allow_html=True)

            # ── 詳細數據展開 ──
            with st.expander(f"🔍 詳細分析 — {info1['cn']} vs {info2['cn']}"):
                det_c1, det_c2 = st.columns(2)

                # 球隊實力對比
                with det_c1:
                    st.markdown("**⚔️ 球隊實力對比（近8年）**")
                    s1, s2 = r['s1'], r['s2']
                    # (label, v1, v2, fmt, direction)
                    # direction: 'high' = 高者較強, 'low' = 低者較強
                    # 平局率為「該隊歷史踢平的傾向」，不適合做兩隊強弱對比 → 移除
                    # 本場平局機率另在卡片右側顯示（單一機率而非雙球隊）
                    metrics_list = [
                        ('勝率',     s1['win_rate'],     s2['win_rate'],     '{:.1%}', 'high'),
                        ('場均進球', s1['avg_goals'],    s2['avg_goals'],    '{:.2f}', 'high'),
                        ('場均失球', s1['avg_conceded'], s2['avg_conceded'], '{:.2f}', 'low'),
                    ]
                    # 傳統高對比配色：標準紅 + 標準藍 + 黃色高亮
                    C1 = '#dc2626'   # 球隊1（標準紅，類似中華隊紅）
                    C2 = '#2563eb'   # 球隊2（標準藍）
                    WIN_BG = 'rgba(250, 204, 21, 0.22)'  # 較強值底色
                    WIN_TX = '#facc15'                    # 較強值文字（金黃）
                    TX = '#ffffff'                        # 普通值（純白）
                    DIM = '#94a3b8'                       # 輸者灰

                    fig_bar = go.Figure()
                    cats = [m[0] for m in metrics_list]
                    v1s = [m[1] for m in metrics_list]
                    v2s = [m[2] for m in metrics_list]
                    fig_bar.add_trace(go.Bar(
                        name=info1['cn'], x=cats, y=v1s,
                        marker=dict(color=C1),
                    ))
                    fig_bar.add_trace(go.Bar(
                        name=info2['cn'], x=cats, y=v2s,
                        marker=dict(color=C2),
                    ))
                    fig_bar.update_layout(
                        barmode='group', height=280,
                        margin=dict(l=0, r=0, t=40, b=40),
                        legend=dict(orientation='h', y=1.15,
                                    font=dict(color='#ffffff', size=14)),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#ffffff', size=13),
                        xaxis=dict(tickfont=dict(color='#ffffff', size=13)),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.12)',
                                   tickfont=dict(color='#e2e8f0', size=12)),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # 純文字標記方案：Streamlit 原生 dataframe + 文字內 ★ 與 ⬇
                    # 不再倚賴 HTML/CSS 配色（會被 Streamlit 過濾），改用 Unicode 符號
                    cn1 = f"🔴 {info1['cn']}"
                    cn2 = f"🔵 {info2['cn']}"
                    cmp_rows = []
                    for label, v1, v2, fmt, direction in metrics_list:
                        # 強制轉 Python bool（避免 numpy bool 與 `is True` 比較失效）
                        if direction == 'high':
                            t1 = bool(v1 > v2); dir_mk = '▲'
                        elif direction == 'low':
                            t1 = bool(v1 < v2); dir_mk = '▼'
                        else:
                            t1 = None; dir_mk = '·'
                        v1_str = fmt.format(v1)
                        v2_str = fmt.format(v2)
                        # 較強者：⭐ 前綴；較弱者：灰色點點前綴；中性：無前綴
                        if t1 is True:
                            v1_str = f"⭐ {v1_str}"
                            v2_str = f"  {v2_str}"
                        elif t1 is False:
                            v1_str = f"  {v1_str}"
                            v2_str = f"⭐ {v2_str}"
                        cmp_rows.append({
                            '指標': f"{dir_mk} {label}",
                            cn1: v1_str,
                            cn2: v2_str,
                        })

                    cmp_df = pd.DataFrame(cmp_rows)
                    st.dataframe(
                        cmp_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            '指標': st.column_config.TextColumn('指標', width='small'),
                            cn1: st.column_config.TextColumn(cn1, width='medium'),
                            cn2: st.column_config.TextColumn(cn2, width='medium'),
                        },
                    )
                    st.caption(
                        f"📊 樣本場數：🔴 {info1['cn']} {s1['matches']} 場  ·  "
                        f"🔵 {info2['cn']} {s2['matches']} 場    "
                        f"｜  ⭐ = 該指標較強  ｜  "
                        f"▲ 高者強 / ▼ 低者強 / · 中性"
                    )

                # Poisson 比分機率熱圖
                with det_c2:
                    st.markdown("**📊 比分機率矩陣（Poisson）**")
                    lam1, lam2 = r['lam1'], r['lam2']
                    mg = 6
                    z = [[float(poisson.pmf(i, lam1) * poisson.pmf(j, lam2)) for j in range(mg)]
                         for i in range(mg)]
                    # highlight predicted score
                    g1p, g2p = min(r['goal1'], mg-1), min(r['goal2'], mg-1)
                    text_z = [[f"{z[i][j]*100:.1f}%{'★' if (i==g1p and j==g2p) else ''}"
                               for j in range(mg)] for i in range(mg)]
                    fig_h = go.Figure(go.Heatmap(
                        z=z, x=[str(j) for j in range(mg)], y=[str(i) for i in range(mg)],
                        colorscale='Blues', showscale=False,
                        text=text_z, texttemplate='%{text}', textfont=dict(size=10),
                    ))
                    fig_h.update_layout(
                        xaxis_title=f'{info2["cn"]} 進球', yaxis_title=f'{info1["cn"]} 進球',
                        height=280, margin=dict(l=40, r=0, t=10, b=40),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#ccc'),
                    )
                    st.plotly_chart(fig_h, use_container_width=True)
                    st.caption(f"★ = 預測最可能比分 {r['goal1']}-{r['goal2']}　λ₁={lam1:.2f} λ₂={lam2:.2f}")

                # 歷史對戰記錄
                st.markdown("**📋 歷史對戰紀錄**")
                h2h = match_df[
                    ((match_df['home_team'] == r['team1']) & (match_df['away_team'] == r['team2'])) |
                    ((match_df['home_team'] == r['team2']) & (match_df['away_team'] == r['team1']))
                ].sort_values('date', ascending=False).head(8).reset_index(drop=True)
                if len(h2h) == 0:
                    st.info("查無歷史對戰紀錄")
                else:
                    h2h_rows = []
                    for _, row in h2h.iterrows():
                        if row['home_team'] == r['team1']:
                            sc = f"{int(row['home_score'])} - {int(row['away_score'])}"
                            winner = info1['cn'] if row['home_score'] > row['away_score'] else (
                                info2['cn'] if row['away_score'] > row['home_score'] else '平局')
                        else:
                            sc = f"{int(row['away_score'])} - {int(row['home_score'])}"
                            winner = info1['cn'] if row['away_score'] > row['home_score'] else (
                                info2['cn'] if row['home_score'] > row['away_score'] else '平局')
                        h2h_rows.append({
                            '日期': str(row['date'])[:10],
                            '賽事': str(row.get('tournament', ''))[:30],
                            f'{info1["cn"]} vs {info2["cn"]}': sc,
                            '勝者': winner,
                        })
                    st.dataframe(pd.DataFrame(h2h_rows), use_container_width=True, hide_index=True)

                # ── TOP5 主將對比 ──
                st.markdown("---")
                st.markdown("**⭐ TOP5 主將能力對比**")
                p1_list = SQUAD_DATA.get(r['team1'], [])
                p2_list = SQUAD_DATA.get(r['team2'], [])
                _pos_order = {'GK': 0, 'CB': 1, 'LB': 1, 'RB': 1, 'CDM': 2, 'CM': 2,
                              'CAM': 3, 'LW': 3, 'RW': 3, 'ST': 4}
                _pos_cn = {'GK': '門將', 'CB': '中後衛', 'LB': '左後衛', 'RB': '右後衛',
                           'CDM': '守備中場', 'CM': '中場', 'CAM': '攻擊中場',
                           'LW': '左翼', 'RW': '右翼', 'ST': '前鋒'}
                p1_list = sorted(p1_list, key=lambda p: _pos_order.get(p['pos'], 5))[:5]
                p2_list = sorted(p2_list, key=lambda p: _pos_order.get(p['pos'], 5))[:5]
                _attr_labels = {'ovr': '綜合', 'pac': '速度', 'sho': '射門',
                                'pas': '傳球', 'dri': '盤帶', 'def': '防守', 'phy': '體能'}
                pc1, pc2 = st.columns(2)
                # 深底用亮色：紅 #ff6b6b、藍 #60a5fa（深底辨識度高）
                for col, plist, color, accent, tinfo in [
                    (pc1, p1_list, '#dc2626', '#ff8a8a', info1),
                    (pc2, p2_list, '#2563eb', '#7aa8ff', info2),
                ]:
                    with col:
                        st.markdown(
                            f"<div style='background:{color};color:#fff;padding:8px 14px;"
                            f"border-radius:6px;font-weight:800;font-size:1rem;margin-bottom:6px'>"
                            f"{tinfo['flag']} {tinfo['cn']}</div>",
                            unsafe_allow_html=True)
                        for p in plist:
                            ovr = p.get('ovr', 0)
                            bar_w = int(ovr)
                            attrs = ' · '.join(
                                f"<span style='color:#cbd5e1'>{lbl}</span>"
                                f"<b style='color:#ffffff;margin-left:2px'>{p.get(k,0)}</b>"
                                for k, lbl in _attr_labels.items() if k != 'ovr'
                            )
                            st.markdown(
                                f"<div style='margin:6px 0;padding:10px 12px;"
                                f"background:rgba(255,255,255,0.08);border-radius:8px;"
                                f"border-left:4px solid {accent}'>"
                                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                                f"<span style='font-weight:700;font-size:0.92rem;color:#ffffff'>{p['name']}"
                                f" <span style='color:{accent};font-size:0.75rem;font-weight:600;margin-left:4px'>"
                                f"{_pos_cn.get(p['pos'], p['pos'])}</span></span>"
                                f"<span style='font-size:1.15rem;font-weight:900;color:{accent}'>{ovr}</span></div>"
                                f"<div style='background:rgba(255,255,255,0.12);border-radius:3px;height:5px;margin:6px 0'>"
                                f"<div style='width:{bar_w}%;height:5px;background:{accent};border-radius:3px'></div></div>"
                                f"<div style='font-size:0.74rem;line-height:1.6'>{attrs}</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        if not plist:
                            st.info("暫無球員資料")

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
                '球隊': t,
                '組別': g,
                '勝率': f"{s['win_rate']:.1%}",
                '場均進球': f"{s['avg_goals']:.2f}",
                '場次': s['matches']
            })

    hist_df = pd.DataFrame(team_hist).sort_values(['組別', '勝率'], ascending=[True, False])
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
        tab_cm, tab_roc, tab_cal, tab_fi, tab_fisher = st.tabs(
            ["📊 混淆矩陣", "📈 ROC 曲線", "🎚 校準曲線", "🔍 特徵重要性", "🧪 費雪檢定"]
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
                title=f"混淆矩陣 — 2022 世界盃（正確率={_acc:.2f}）",
                xaxis_title="預測結果", yaxis_title="實際結果",
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
                title="ROC 曲線 — 一對多（2022 世界盃測試集）",
                xaxis_title="假陽性率",
                yaxis_title="真陽性率",
                height=420,
                legend=dict(x=0.62, y=0.08),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.caption("AUC > 0.7 表示模型對各類別有明顯鑑別力；平局 AUC 最低，符合足球平局難預測的直覺。")

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
                title="校準曲線 — 主隊勝類別",
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
                title="特徵重要性（XGBoost gain）",
                xaxis_title="重要性分數",
                height=max(350, len(fi['features']) * 28),
                margin=dict(l=160),
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            st.caption("數值越高代表該特徵對模型決策影響越大。FIFA 排名差距（rank_diff）通常是最強預測因子。")

        # ── Tab 5: Fisher's Exact Test ──
        with tab_fisher:
            from scipy.stats import fisher_exact, chi2_contingency
            cm_arr = np.array(em['cm'])
            total_n = int(cm_arr.sum())
            class_cn = {'Team1 Lose': '主隊負', 'Draw': '平局', 'Team1 Win': '主隊勝'}

            st.markdown(
                "**統計檢定方法：**「費雪精確檢定」（Fisher's Exact Test）— "
                "用 **一對多（One-vs-Rest）** 把 3×3 混淆矩陣拆成三組 2×2，"
                "檢定模型「預測該類別」與「實際發生該類別」是否有顯著關聯。"
            )

            fisher_rows = []
            for c, label in enumerate(em['labels_name']):
                tp = int(cm_arr[c, c])
                fn = int(cm_arr[c, :].sum() - tp)
                fp = int(cm_arr[:, c].sum() - tp)
                tn = int(total_n - tp - fn - fp)
                table_2x2 = [[tp, fn], [fp, tn]]
                try:
                    odds_ratio, p_value = fisher_exact(table_2x2, alternative='greater')
                except Exception:
                    odds_ratio, p_value = float('nan'), float('nan')
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fisher_rows.append({
                    '類別': class_cn.get(label, label),
                    'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                    'Precision': f"{precision:.2%}",
                    'Recall': f"{recall:.2%}",
                    'Odds Ratio': f"{odds_ratio:.2f}" if not np.isnan(odds_ratio) else '—',
                    'p-value': f"{p_value:.4f}" if not np.isnan(p_value) else '—',
                    '顯著性 (α=0.05)': '✅ 顯著優於隨機' if (not np.isnan(p_value) and p_value < 0.05) else '⚠️ 未達顯著',
                })
            fisher_df = pd.DataFrame(fisher_rows)
            st.dataframe(fisher_df, use_container_width=True, hide_index=True)

            # 全表卡方檢定（3×3 整體獨立性）
            try:
                chi2, chi_p, dof, _ = chi2_contingency(cm_arr)
                chi2_ok = chi_p < 0.05
            except Exception:
                chi2, chi_p, dof, chi2_ok = float('nan'), float('nan'), 0, False

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("整體卡方統計量 χ²", f"{chi2:.2f}" if not np.isnan(chi2) else '—',
                          help="3×3 整體獨立性檢定（卡方近似），補充費雪一對多分析")
            with col_b:
                st.metric("自由度 (df)", str(dof))
            with col_c:
                st.metric("p-value", f"{chi_p:.4f}" if not np.isnan(chi_p) else '—',
                          delta="顯著" if chi2_ok else "未達顯著",
                          delta_color="normal" if chi2_ok else "off")

            st.caption(
                "📖 **判讀方式**：p-value < 0.05 表示在統計上拒絕「模型預測與實際結果獨立」的虛無假設，"
                "代表該類別模型的預測能力顯著優於隨機。Odds Ratio 越大代表正相關越強（>1 = 預測有效）。"
                "平局類別 p-value 通常較大，符合足球平局難以預測的直覺。"
            )

# ============================================================
# PAGE 4: 各國分析 + 球員能力卡
# ============================================================
elif page == "🌍 各國分析":
    from squad_data import SQUAD_DATA

    st.title("🌍 各國深度分析")
    st.markdown("**選擇一支球隊，查看歷史數據分析與本屆出賽球員能力卡**")
    st.markdown("---")

    match_df = load_match_data()
    fifa_df  = load_fifa_ranking()

    # ── 球員卡 CSS（FIFA 風格）──
    st.markdown("""
    <style>
    .player-card {
        background: linear-gradient(145deg, #1a2a4a 0%, #0d1b2e 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 16px 14px 14px;
        margin-bottom: 12px;
        text-align: center;
        box-shadow: 0 4px 18px rgba(0,0,0,0.4);
        min-height: 240px;
    }
    .pc-ovr  { font-size: 2.4rem; font-weight: 900; color: #f7c948; line-height: 1; }
    .pc-pos  { font-size: 0.75rem; font-weight: 700; color: #aabbcc; letter-spacing: 1px; margin-bottom: 4px; }
    .pc-name { font-size: 0.92rem; font-weight: 700; color: #ffffff; margin: 6px 0 4px; }
    .pc-club { font-size: 0.72rem; color: #7a9ab5; margin-bottom: 10px; }
    .pc-flag { font-size: 1.6rem; margin-bottom: 2px; }
    .pc-attr-row { display: flex; justify-content: space-between; font-size: 0.68rem;
                   color: #ccd; margin: 2px 0; }
    .pc-attr-label { color: #8899aa; font-weight: 600; }
    .pc-attr-bar-wrap { flex: 1; margin: 0 6px; background: rgba(255,255,255,0.08);
                        border-radius: 4px; height: 7px; margin-top: 3px; }
    .pc-attr-bar { height: 7px; border-radius: 4px; }
    .divider { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)

    # ── 國家選擇器 ──
    all_teams_sorted = sorted(TEAM_INFO.keys())
    group_map = {t: g for g, teams in WC_2026_GROUPS.items() for t in teams}
    team_options = [t for t in all_teams_sorted if t in group_map]

    selected_team = st.selectbox(
        "選擇國家",
        team_options,
        format_func=lambda t: f"{TEAM_INFO[t]['flag']} {TEAM_INFO[t]['cn']} ({TEAM_INFO[t]['en']}) — Group {group_map.get(t,'?')}"
    )

    info = TEAM_INFO[selected_team]
    iso  = info.get('iso', 'un')
    grp  = group_map.get(selected_team, '?')

    # ── 國家 Header ──
    col_flag, col_info = st.columns([1, 4])
    with col_flag:
        st.markdown(f'<img src="https://flagcdn.com/80x60/{iso}.png" style="border-radius:6px;width:100%;max-width:110px;">',
                    unsafe_allow_html=True)
    with col_info:
        st.markdown(f"## {info['cn']}（{info['en']}）")
        st.markdown(f"**FIFA 排名：#{info['fifa_rank']}　FIFA 積分：{info['fifa_pts']}　所在小組：Group {grp}**")
        conf = next((v for k, v in {
            'France':'UEFA','Spain':'UEFA','England':'UEFA','Germany':'UEFA','Portugal':'UEFA',
            'Netherlands':'UEFA','Belgium':'UEFA','Croatia':'UEFA','Switzerland':'UEFA',
            'Poland':'UEFA','Sweden':'UEFA','Austria':'UEFA','Czechia':'UEFA','Scotland':'UEFA',
            'Norway':'UEFA','Bosnia and Herzegovina':'UEFA','Turkiye':'UEFA',
            'Brazil':'CONMEBOL','Argentina':'CONMEBOL','Uruguay':'CONMEBOL','Colombia':'CONMEBOL',
            'Paraguay':'CONMEBOL','Ecuador':'CONMEBOL',
            'USA':'CONCACAF','Mexico':'CONCACAF','Canada':'CONCACAF','Panama':'CONCACAF',
            'Curacao':'CONCACAF','Haiti':'CONCACAF',
            'Japan':'AFC','South Korea':'AFC','Iran':'AFC','Australia':'AFC',
            'Saudi Arabia':'AFC','Iraq':'AFC','Jordan':'AFC','Uzbekistan':'AFC','Qatar':'AFC',
            'New Zealand':'OFC',
        }.items() if k == selected_team), 'CAF')
        st.markdown(f"**洲際聯盟：{conf}**")

    st.markdown("---")

    # ── 歷史統計 + 近期表現 ──
    s = compute_team_strength(match_df, selected_team, 2026)
    recent_matches = match_df[
        ((match_df['home_team'] == selected_team) | (match_df['away_team'] == selected_team)) &
        (match_df['year'] >= 2022)
    ].sort_values('date', ascending=False).head(8)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1: st.metric("近 8 年勝率", f"{s['win_rate']:.1%}")
    with col_s2: st.metric("平均進球", f"{s['avg_goals']:.2f}")
    with col_s3: st.metric("平均失球", f"{s['avg_conceded']:.2f}")
    with col_s4: st.metric("出賽場數", f"{s['matches']}")

    st.markdown("---")

    # ── 近期賽果 + 小組賽程 ──
    tab_recent, tab_group, tab_players = st.tabs(["📅 近期賽果", "⚔️ 小組賽程預測", "👤 球員能力卡"])

    with tab_recent:
        st.markdown("##### 2022年後近期賽果（最多 8 場）")
        if len(recent_matches) == 0:
            st.info("無近期賽事資料")
        else:
            for _, row in recent_matches.iterrows():
                is_home = row['home_team'] == selected_team
                opp = row['away_team'] if is_home else row['home_team']
                opp_info = TEAM_INFO.get(opp, {'flag':'🏳️','cn':opp,'iso':'un'})
                gf = row['home_score'] if is_home else row['away_score']
                ga = row['away_score'] if is_home else row['home_score']
                result = "✅ 勝" if gf > ga else ("❌ 負" if gf < ga else "🟡 平")
                venue = "主場" if is_home else "客場"
                opp_iso = opp_info.get('iso','un')
                opp_flag_html = f'<img src="https://flagcdn.com/20x15/{opp_iso}.png" style="vertical-align:middle;margin-right:4px;border-radius:2px;">'
                st.markdown(
                    f"{result} &nbsp; **{row['date'].strftime('%Y-%m-%d')}** &nbsp; "
                    f"（{venue}）vs {opp_flag_html}{opp_info['cn']} &nbsp; **{int(gf)} - {int(ga)}** &nbsp; "
                    f"<span style='color:#7a9ab5;font-size:0.8rem;'>{row['tournament']}</span>",
                    unsafe_allow_html=True
                )

    with tab_group:
        group_teams = WC_2026_GROUPS[grp]
        opponents = [t for t in group_teams if t != selected_team]
        pre = load_pretrained()
        if pre:
            clf, p1, p2, fc = pre['clf'], pre['poisson1'], pre['poisson2'], pre['feat_cols']
            st.markdown(f"##### Group {grp} 小組賽預測")
            for opp in opponents:
                opp_info = TEAM_INFO.get(opp, {'flag':'🏳️','cn':opp,'iso':'un'})
                opp_iso = opp_info.get('iso','un')
                pred = predict_match(selected_team, opp, 2026, match_df, fifa_df, clf, p1, p2, fc)
                my_iso = info.get('iso','un')
                my_flag_img = f'<img src="https://flagcdn.com/24x18/{my_iso}.png" style="border-radius:2px;vertical-align:middle;">'
                opp_flag_img = f'<img src="https://flagcdn.com/24x18/{opp_iso}.png" style="border-radius:2px;vertical-align:middle;">'
                # 依比分決定顏色，確保與顯示結果一致
                sc1 = '#00d4ff' if pred['goal1'] > pred['goal2'] else ('#8899aa' if pred['goal1'] < pred['goal2'] else '#f7c948')
                sc2 = '#00d4ff' if pred['goal1'] < pred['goal2'] else ('#8899aa' if pred['goal1'] > pred['goal2'] else '#f7c948')
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:10px 16px;margin-bottom:8px;display:flex;align-items:center;gap:10px;">
                  <span style="display:flex;align-items:center;gap:6px;">{my_flag_img}<b>{info['cn']}</b>
                    <span style="font-size:1.4rem;font-weight:900;color:{sc1}">{pred['goal1']}</span></span>
                  <span style="color:#8899aa;font-size:0.85rem;">VS</span>
                  <span style="display:flex;align-items:center;gap:6px;">
                    <span style="font-size:1.4rem;font-weight:900;color:{sc2}">{pred['goal2']}</span>
                    <b>{opp_info['cn']}</b>{opp_flag_img}</span>
                  <span style="margin-left:auto;color:#aabbcc;font-size:0.82rem;">平局 <b style="color:#f7c948">{pred['draw_prob']:.0%}</b></span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("請先執行 python pretrain.py 產生模型")

    with tab_players:
        players = SQUAD_DATA.get(selected_team, [])
        if not players:
            st.info("此隊球員資料尚未收錄")
        else:
            st.markdown(f"##### {info['cn']} 出賽球員能力卡（共 {len(players)} 人）")
            import streamlit.components.v1 as _components

            ATTR_COLORS = {
                'pac': '#00d4ff', 'sho': '#e94560', 'pas': '#26de81',
                'dri': '#f7c948', 'def': '#a29bfe', 'phy': '#fd9644',
            }
            ATTR_LABELS = {'pac':'PAC','sho':'SHO','pas':'PAS','dri':'DRI','def':'DEF','phy':'PHY'}

            # sofifa ID lookup for well-known players
            SOFIFA_IDS = {
                'L. Messi': 158023, 'C. Ronaldo': 20801, 'K. Mbappé': 231747,
                'E. Haaland': 239085, 'K. De Bruyne': 192985, 'Vinícius Jr.': 246169,
                'J. Bellingham': 253072, 'M. Salah': 209331, 'L. Modrić': 177003,
                'Son Heung-min': 202126, 'K. Havertz': 246669, 'F. Wirtz': 263552,
                'R. Leão': 251706, 'A. Davies': 236460, 'J. David': 254103,
                'M. Ødegaard': 231568, 'A. Hakimi': 238023, 'V. van Dijk': 203376,
                'L. Díaz': 243726, 'A. Griezmann': 194765, 'R. Mahrez': 220054,
                'S. Mané': 208722, 'T. Partey': 209564, 'M. Kudus': 260460,
                'Pedri': 252371, 'L. Yamal': 272456, 'F. de Jong': 239473,
                'F. Valverde': 241764, 'D. Núñez': 261204, 'J. Álvarez': 261289,
                'B. Fernandes': 212831, 'R. Dias': 237692, 'H. Kane': 202126,
                'G. Xhaka': 186942, 'M. Akanji': 231678, 'J. Kimmich': 214455,
                'Alisson': 211110, 'Marquinhos': 200389, 'Rodrygo': 252658,
                'V. Gyökeres': 243726, 'D. Kulusevski': 246940, 'M. Neuer': 167495,
                'A. Rüdiger': 206158, 'A. Robertson': 225321, 'J. Gvardiol': 261255,
            }

            # 整批建一個 HTML，用 CSS Grid 排列，交給 components.html 完整渲染
            cards_html = ""
            for p in players:
                ovr = p['ovr']
                ovr_color = '#f7c948' if ovr >= 85 else ('#00d4ff' if ovr >= 78 else '#aaaacc')
                attrs_rows = ""
                for attr_key in ['pac','sho','pas','dri','def','phy']:
                    val = p[attr_key]
                    bc  = ATTR_COLORS[attr_key]
                    lbl = ATTR_LABELS[attr_key]
                    attrs_rows += (
                        f'<div style="display:flex;align-items:center;margin:3px 0;font-size:0.7rem;">'
                        f'<span style="color:#8899aa;font-weight:700;width:28px;">{lbl}</span>'
                        f'<div style="flex:1;background:rgba(255,255,255,0.1);border-radius:4px;height:7px;margin:0 6px;">'
                        f'<div style="width:{val}%;height:7px;border-radius:4px;background:{bc};"></div></div>'
                        f'<span style="color:#e8e8e8;font-weight:700;width:22px;text-align:right;">{val}</span>'
                        f'</div>'
                    )
                # player photo: sofifa CDN with initials fallback
                sid = SOFIFA_IDS.get(p['name'], 0)
                initials = ''.join(w[0] for w in p['name'].replace('.','').split() if w)[:2].upper()
                fallback_url = f"https://ui-avatars.com/api/?name={initials}&background=1a2a4a&color=f7c948&size=80&bold=true&rounded=true&length=2"
                if sid:
                    photo_src = f"https://cdn.sofifa.net/players/{sid}/26_60.png"
                    photo_html = (
                        f'<img src="{photo_src}" '
                        f'onerror="this.onerror=null;this.src=\'{fallback_url}\';" '
                        f'style="width:64px;height:64px;border-radius:50%;object-fit:cover;'
                        f'border:2px solid rgba(247,201,72,0.4);margin-bottom:6px;">'
                    )
                else:
                    photo_html = (
                        f'<img src="{fallback_url}" '
                        f'style="width:64px;height:64px;border-radius:50%;object-fit:cover;'
                        f'border:2px solid rgba(255,255,255,0.15);margin-bottom:6px;">'
                    )
                cards_html += (
                    f'<div style="background:linear-gradient(145deg,#1a2a4a,#0d1b2e);border:1px solid rgba(255,255,255,0.1);'
                    f'border-radius:14px;padding:16px 14px;text-align:center;box-shadow:0 4px 18px rgba(0,0,0,0.4);">'
                    f'{photo_html}'
                    f'<div style="font-size:2rem;font-weight:900;color:{ovr_color};line-height:1.1;">{ovr}</div>'
                    f'<div style="font-size:0.72rem;font-weight:700;color:#aabbcc;letter-spacing:1px;">{p["pos"]}</div>'
                    f'<hr style="border:none;border-top:1px solid rgba(255,255,255,0.1);margin:6px 0;">'
                    f'<div style="font-size:0.88rem;font-weight:700;color:#fff;">{p["name"]}</div>'
                    f'<div style="font-size:0.7rem;color:#7a9ab5;margin-bottom:8px;">{p["club"]} · {p["age"]}歲</div>'
                    f'<hr style="border:none;border-top:1px solid rgba(255,255,255,0.1);margin:6px 0;">'
                    f'{attrs_rows}'
                    f'</div>'
                )

            full_html = (
                f'<div style="display:grid;grid-template-columns:repeat({min(len(players),5)},1fr);gap:12px;">'
                f'{cards_html}</div>'
            )
            _components.html(full_html, height=460, scrolling=False)

# ============================================================
# PAGE 5: 球隊風格分群（v2.2 新增 — 滿足課程「分群」任務）
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
        plot_df['hover_name'] = plot_df['team_cn']   # 純文字，避免 emoji 亂碼

        # Plotly 圖例用 ASCII 標籤（避免 CJK+emoji 亂碼）
        en_labels = {i: clusters.get('cluster_names_en', {}).get(i, f'Group {i+1}') for i in range(k)}
        plot_df['style_en'] = plot_df['cluster'].map(en_labels)

        fig_pca = px.scatter(
            plot_df, x='pca1', y='pca2',
            color='style_en',
            hover_name='hover_name',
            hover_data={'style_en': True, 'cluster_label': True, 'pca1': False, 'pca2': False},
            text='team_cn',
            title=f'球隊風格分群 — PCA 二維投影  (k={k}, Silhouette={sil:.2f})',
            color_discrete_sequence=['#e94560', '#f5a623', '#3366cc', '#0f6e6e'],
            height=540,
        )
        fig_pca.update_traces(textposition='top center', marker=dict(size=11),
                              textfont=dict(size=9))
        fig_pca.update_layout(
            legend_title_text='風格',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ccc'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
        )
        st.plotly_chart(fig_pca, use_container_width=True)

        # ── 分群說明：類型定義 + 各群中心數據 ──
        style_colors = {'攻擊型': '#e94560', '防守型': '#3366cc', '平衡型': '#f5a623'}
        style_icons  = {'攻擊型': '⚡', '防守型': '🛡️', '平衡型': '⚖️'}
        style_criteria = {
            '攻擊型': '場均進球最高 → 主動進攻、以攻代守',
            '防守型': '場均失球最低 → 穩守反擊、嚴密組織',
            '平衡型': '攻守均衡 → 彈性戰術、視對手調整',
        }
        legend_cols = st.columns(k)
        tier_colors_list = ['#e94560', '#f5a623', '#3366cc', '#0f6e6e']
        for i in range(k):
            cname_full = cluster_names.get(i, f'Group {i+1}')
            # 取出純中文名（去 emoji）
            cn_pure = cname_full.replace('⚡','').replace('🛡️','').replace('⚖️','').strip()
            color = style_colors.get(cn_pure, tier_colors_list[i % len(tier_colors_list)])
            icon  = style_icons.get(cn_pure, '')
            criteria = style_criteria.get(cn_pure, '')
            # 各群中心攻守數據
            c_row = centers.iloc[i] if hasattr(centers, 'iloc') else {}
            goals_c    = c_row.get('avg_goals', 0) if isinstance(c_row, dict) else float(c_row['avg_goals'])
            conceded_c = c_row.get('avg_conceded', 0) if isinstance(c_row, dict) else float(c_row['avg_conceded'])
            with legend_cols[i]:
                st.markdown(
                    f'<div style="border-left:4px solid {color};padding:8px 12px;'
                    f'border-radius:4px;background:rgba(255,255,255,0.04);margin-bottom:6px;">'
                    f'<b style="font-size:1rem;">{icon} {cn_pure}</b><br>'
                    f'<span style="color:#aaa;font-size:0.8rem;">{criteria}</span><br>'
                    f'<span style="font-size:0.8rem;">場均進球 <b style="color:#00d4ff">{goals_c:.2f}</b> ／ '
                    f'場均失球 <b style="color:#f7c948">{conceded_c:.2f}</b></span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

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
        for cid, cname in sorted(cluster_names.items()):
            cn_pure = cname.replace('⚡','').replace('🛡️','').replace('⚖️','').strip()
            color = style_colors.get(cn_pure, '#aaa')
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

        feat_cols_radar = [c for c in df_c.columns if c not in ('cluster', 'pca1', 'pca2')]
        _radar_label_map = {
            'win_rate': '勝率', 'avg_goals': '場均進球', 'avg_conceded': '場均失球',
            'goal_diff': '淨勝球', 'matches': '出賽場數',
            'knockout_exp': '淘汰賽經驗', 'fifa_pts': 'FIFA積分',
        }
        radar_labels = [_radar_label_map.get(c, c) for c in feat_cols_radar]
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
                r=norm_a + [norm_a[0]], theta=radar_labels + [radar_labels[0]],
                fill='toself', name=TEAM_INFO.get(team_a, {'cn': team_a})['cn'],
                line_color='#e94560',
            ))
            fig_radar.add_trace(_go_r.Scatterpolar(
                r=norm_b + [norm_b[0]], theta=radar_labels + [radar_labels[0]],
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
        n_sims = mc.get('n_sims', 10000)
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
                st.markdown(f'<div class="metric-card"><h2>{top["flag_cn"]}</h2><p>奪冠熱門 {top["champ_pct"]:.1f}%</p></div>', unsafe_allow_html=True)
            if len(mc_df) > 1:
                with c2:
                    t2 = mc_df.iloc[1]
                    st.markdown(f'<div class="metric-card"><h2>{t2["flag_cn"]}</h2><p>第二熱門 {t2["champ_pct"]:.1f}%</p></div>', unsafe_allow_html=True)
            if len(mc_df) > 2:
                with c3:
                    t3 = mc_df.iloc[2]
                    st.markdown(f'<div class="metric-card"><h2>{t3["flag_cn"]}</h2><p>第三熱門 {t3["champ_pct"]:.1f}%</p></div>', unsafe_allow_html=True)

        st.markdown("---")
        top20 = mc_df.head(20)
        fig_mc = px.bar(
            top20, x='champ_pct', y='flag_cn',
            orientation='h',
            title=f'奪冠機率 Top 20（Monte Carlo {n_sims:,} 次模擬）',
            labels={'champ_pct': '奪冠機率', 'flag_cn': '球隊'},
            color='champ_pct',
            color_continuous_scale='Reds',
            text=top20['champ_pct'].apply(lambda x: f'{x:.1f}%'),
        )
        fig_mc.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis=dict(range=[0, 100], ticksuffix='%'),
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
            display_df[col] = display_df[col].apply(lambda x: f'{x:.1f}%')
        st.dataframe(display_df, use_container_width=True)

# ============================================================
# PAGE 6: 完整賽程
# ============================================================
elif page == "📅 完整賽程":
    st.title("📅 2026 世界盃淘汰賽對陣圖")
    st.caption("48 隊 · 🏆決賽（頂端）向下展開至32強 · 時間為台灣時間（UTC+8）· 2026年7月")

    import streamlit.components.v1 as _c

    # ── 版面參數（由上而下樹狀圖：決賽在頂，32強在底）──
    BW    = 54     # match box 寬度
    BHB   = 40     # match box 高度
    UNIT  = 60     # BW + 6px 間距
    N32   = 16     # 32強場數
    LM    = 34     # 左邊留給輪次標籤的空間
    SC    = 'rgba(100,160,220,0.7)'   # 連接線顏色
    RSTEP = 82     # 各輪 Y 中心間距

    # ── 各輪 X 中心（由底部 32強 往上推算）──
    xc5 = [LM + i * UNIT + BW // 2 for i in range(N32)]
    xc4 = [(xc5[i*2] + xc5[i*2+1]) // 2 for i in range(8)]
    xc3 = [(xc4[i*2] + xc4[i*2+1]) // 2 for i in range(4)]
    xc2 = [(xc3[i*2] + xc3[i*2+1]) // 2 for i in range(2)]
    xc1 = [(xc2[0] + xc2[1]) // 2]
    TW   = LM + (N32 - 1) * UNIT + BW + 4

    # ── 各輪 Y 中心（頂→底：決賽→四強→八強→16強→32強）──
    YC = [BHB // 2 + i * RSTEP for i in range(5)]
    # YC = [23, 123, 223, 323, 423]
    CANVAS_H = YC[4] + BHB // 2 + 14

    # ── 連接點 Y（兩輪中間）──
    JY = [(YC[i] + YC[i+1]) // 2 for i in range(4)]
    # JY = [73, 173, 273, 373]

    # ── Match box HTML（絕對定位）──
    def mb(xc: int, yc: int, ts: str, t1: str, t2: str, highlight: bool = False) -> str:
        x   = xc - BW // 2
        y   = yc - BHB // 2
        bg  = 'rgba(247,201,72,0.12)' if highlight else '#0c1c30'
        bdr = 'rgba(247,201,72,0.7)'  if highlight else 'rgba(100,150,200,0.28)'
        return (
            f'<div style="position:absolute;left:{x}px;top:{y}px;width:{BW}px;height:{BHB}px;'
            f'border:1px solid {bdr};border-radius:4px;background:{bg};'
            f'padding:3px 5px;box-sizing:border-box;overflow:hidden;">'
            f'<div style="color:#4a7ea8;font-size:0.5rem;line-height:1.4;">{ts}</div>'
            f'<div style="color:#c8e0f4;font-weight:700;font-size:0.57rem;white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis;">{t1}</div>'
            f'<div style="color:#c8e0f4;font-weight:700;font-size:0.57rem;white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis;">VS {t2}</div>'
            f'</div>'
        )

    # ── SVG 線段 ──
    def vl(x, y1, y2): return f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{SC}" stroke-width="1.5"/>'
    def hl(x1, x2, y): return f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{SC}" stroke-width="1.5"/>'

    def connect(par_xcs, chi_xcs, par_lvl):
        """由父輪往下連接到子輪（頂→底方向）"""
        lines = ""
        j = JY[par_lvl]
        yp = YC[par_lvl]
        yc = YC[par_lvl + 1]
        for pi, px in enumerate(par_xcs):
            cl, cr = chi_xcs[pi * 2], chi_xcs[pi * 2 + 1]
            lines += vl(px, yp + BHB // 2, j)      # 父底部 → 連接點
            lines += hl(cl, cr, j)                   # 橫槓連左右子
            lines += vl(cl, j, yc - BHB // 2)        # 連接點 → 左子頂部
            lines += vl(cr, j, yc - BHB // 2)        # 連接點 → 右子頂部
        return lines

    # ── 比賽資料（32強：8場左 + 8場右，共16場；各輪配對往下對應）──
    r32_all = [
        ("7/4 09:00","A組1","B組2"),    ("7/4 22:00","C組1","D組2"),
        ("7/5 09:00","E組1","F組2"),    ("7/5 22:00","G組1","H組2"),
        ("7/6 09:00","I組1","J組2"),    ("7/6 22:00","K組1","L組2"),
        ("7/7 09:00","最佳3rd①","最佳3rd②"), ("7/7 22:00","最佳3rd③","最佳3rd④"),
        ("7/4 13:00","B組1","A組2"),    ("7/4 16:00","D組1","C組2"),
        ("7/5 13:00","F組1","E組2"),    ("7/5 16:00","H組1","G組2"),
        ("7/6 13:00","J組1","I組2"),    ("7/6 16:00","L組1","K組2"),
        ("7/7 13:00","最佳3rd⑤","最佳3rd⑥"), ("7/7 16:00","最佳3rd⑦","最佳3rd⑧"),
    ]
    r16_all = [
        ("7/9 09:00","R32①勝","R32②勝"),  ("7/9 22:00","R32③勝","R32④勝"),
        ("7/10 09:00","R32⑤勝","R32⑥勝"), ("7/10 22:00","R32⑦勝","R32⑧勝"),
        ("7/11 09:00","R32⑨勝","R32⑩勝"), ("7/11 22:00","R32⑪勝","R32⑫勝"),
        ("7/12 09:00","R32⑬勝","R32⑭勝"), ("7/12 22:00","R32⑮勝","R32⑯勝"),
    ]
    qf_all = [
        ("7/14 09:00","16強①勝","16強②勝"), ("7/14 22:00","16強③勝","16強④勝"),
        ("7/15 09:00","16強⑤勝","16強⑥勝"), ("7/15 22:00","16強⑦勝","16強⑧勝"),
    ]
    sf_all = [
        ("7/18 22:00","八強①勝","八強②勝"),
        ("7/19 22:00","八強③勝","八強④勝"),
    ]
    fin_d = ("7/22 21:00", "四強①勝", "四強②勝")

    # ── 組合 HTML ──
    body = f'<div style="position:relative;width:{TW}px;height:{CANVAS_H}px;">'

    for i, m in enumerate(r32_all): body += mb(xc5[i], YC[4], m[0], m[1], m[2])
    for i, m in enumerate(r16_all): body += mb(xc4[i], YC[3], m[0], m[1], m[2])
    for i, m in enumerate(qf_all):  body += mb(xc3[i], YC[2], m[0], m[1], m[2])
    for i, m in enumerate(sf_all):  body += mb(xc2[i], YC[1], m[0], m[1], m[2])
    body += mb(xc1[0], YC[0], fin_d[0], fin_d[1], fin_d[2], highlight=True)

    # SVG（連接線 + 輪次標籤）
    svg = (f'<svg style="position:absolute;top:0;left:0;width:{TW}px;height:{CANVAS_H}px;'
           f'pointer-events:none;" xmlns="http://www.w3.org/2000/svg">')
    svg += connect(xc1, xc2, 0)   # 決賽 → 四強
    svg += connect(xc2, xc3, 1)   # 四強 → 八強
    svg += connect(xc3, xc4, 2)   # 八強 → 16強
    svg += connect(xc4, xc5, 3)   # 16強 → 32強

    # 輪次標籤（SVG text，靠左對齊）
    for lvl_i, (label, color) in enumerate([
        ("🏆決賽", "#f7c948"), ("四強", "#c8e0f4"),
        ("八強",   "#c8e0f4"), ("16強", "#c8e0f4"), ("32強", "#c8e0f4"),
    ]):
        svg += (f'<text x="{LM - 4}" y="{YC[lvl_i] + 5}" text-anchor="end" '
                f'font-size="11" font-weight="bold" fill="{color}" '
                f'font-family="Noto Sans TC,sans-serif">{label}</text>')

    svg += '</svg>'
    body += svg + '</div>'

    third_bar = (
        f'<div style="margin-top:8px;padding:5px 10px;border:1px solid rgba(200,170,60,0.28);'
        f'border-radius:5px;background:rgba(200,170,60,0.04);">'
        f'<span style="color:#c8a850;font-weight:700;font-size:0.68rem;">🥉 季軍賽</span>'
        f'<span style="color:#6a8fb0;font-size:0.63rem;margin-left:10px;">'
        f'7/22 04:00 · 四強負方① VS 四強負方②</span>'
        f'</div>'
    )

    full_html = (
        f'<div style="background:#091525;border-radius:12px;padding:10px 12px;'
        f'font-family:\'Noto Sans TC\',sans-serif;overflow-x:auto;min-width:{TW}px;">'
        f'{body}{third_bar}'
        f'<div style="text-align:center;color:#1a3050;font-size:0.55rem;margin-top:5px;">'
        f'※ 對陣組合以 FIFA 官方公布為準 · 時間為台灣時間（UTC+8）</div>'
        f'</div>'
    )

    st.caption(f"賽程總寬度 {TW}px · 若未完整顯示可左右滑動")
    _c.html(full_html, height=CANVAS_H + 85, scrolling=True)

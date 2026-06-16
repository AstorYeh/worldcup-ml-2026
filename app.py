"""
🏆 2026 World Cup ML Prediction System — Streamlit 主程式
============================================================

五層架構（四層預測管線 + 市場校準）：
    L1  XGBoost 分類器        (外層權重 60%)
    L2  Dixon-Coles Poisson  (外層權重 40%)
    L3  λ 多因素複合修正     (主將/FIFA/狀態/經驗)
    L4  融合機率 + MC 10,000 次（MC 含主將 OVR 修正）
    L5  市場校準（奪冠頁）   最終=(1−α)模型+α市場，α 隨開賽浮動

7 個分頁：
    📊 專題總覽 · 🔮 2026 預測 · 📈 數據分析 · 🌍 各國分析
    🎯 球隊風格分群 · 🏅 奪冠預測 · 📅 完整賽程

執行：
    1) python pretrain.py   ← 必須先跑一次，產出 models/*.pkl
    2) streamlit run app.py
"""

import os
import base64
import streamlit as st
import streamlit.components.v1 as components
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
# 單一資料來源：48 隊 FIFA 資料、分組、足聯（與 pretrain.py 共用，避免漂移）
from constants import TEAM_INFO, WC_2026_GROUPS, CONFED_MAP, CONFED_BONUS
# 市場情緒量化：博彩賠率去抽水後的隱含奪冠機率（奪冠頁市場校準層用）
import market_odds as MARKET
from datetime import date as _date

# ============================================================
# Plotly 全域 Claude 主題（暖色淺底 + 珊瑚橘）
# ============================================================
_CLAUDE_BG    = '#faf9f5'
_CLAUDE_CARD  = '#ffffff'
_CLAUDE_TEXT  = '#1f1e1c'
_CLAUDE_MUTE  = '#6b6760'
_CLAUDE_GRID  = '#e8e5dd'
_CLAUDE_ACCENT = '#c96442'
_CLAUDE_ACCENT2 = '#1f6e8c'
_CLAUDE_PALETTE = ['#c96442', '#1f6e8c', '#b58a3b', '#5f8466',
                   '#7a5fa7', '#a8593e', '#3d6e8a', '#8e6b3f']


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """轉 #RRGGBB → rgba(r,g,b,alpha) 字串供 plotly fillcolor 使用。"""
    h = hex_color.lstrip('#')
    if len(h) != 6:
        return f"rgba(201,100,66,{alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def claude_layout(**overrides):
    """回傳 Claude 主題的 plotly layout dict（暖色淺底）。
    注意：title 欄位不在 base 內，避免 None text 被渲染為 "undefined"。
    如需設標題，用 overrides 傳入 title=dict(text='...')。
    """
    base = dict(
        paper_bgcolor=_CLAUDE_CARD,
        plot_bgcolor=_CLAUDE_CARD,
        font=dict(family='Inter, system-ui, sans-serif',
                  color=_CLAUDE_TEXT, size=12),
        xaxis=dict(gridcolor=_CLAUDE_GRID, linecolor=_CLAUDE_GRID,
                   tickcolor=_CLAUDE_GRID, color=_CLAUDE_TEXT, zeroline=False),
        yaxis=dict(gridcolor=_CLAUDE_GRID, linecolor=_CLAUDE_GRID,
                   tickcolor=_CLAUDE_GRID, color=_CLAUDE_TEXT, zeroline=False),
        legend=dict(font=dict(color=_CLAUDE_TEXT)),
        colorway=_CLAUDE_PALETTE,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    base.update(overrides)
    return base

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
/* ============================================================
   ── Theme: Claude-inspired (warm cream + coral) ──
   參考 Anthropic claude.ai 設計語言：暖色奶油底、珊瑚橘強調、
   襯線標題、極簡邊框、低調陰影
============================================================ */
:root {
    --bg-primary:    #faf9f5;       /* 奶油色主背景 */
    --bg-card:       #ffffff;       /* 卡片純白 */
    --bg-elevated:   #f4f1ea;       /* 微抬升底色 */
    --bg-subtle:     #ede9df;       /* 更深一層底色 */
    --accent:        #c96442;       /* Anthropic 珊瑚橘 */
    --accent-hover:  #b85838;       /* 深一階珊瑚橘 */
    --accent-soft:   #f3e9e4;       /* 珊瑚橘微透 */
    --accent-2:      #1f6e8c;       /* 輔助靛藍 */
    --text-primary:  #1f1e1c;       /* 暖近黑 */
    --text-secondary:#6b6760;       /* 中灰 */
    --text-tertiary: #9b958a;       /* 淺灰 */
    --border:        #e8e5dd;       /* 柔邊框 */
    --border-strong: #d4d1c8;       /* 重邊框 */
    --shadow-sm:     0 1px 2px rgba(31,30,28,0.04);
    --shadow-md:     0 2px 8px rgba(31,30,28,0.06);
    --shadow-lg:     0 4px 16px rgba(31,30,28,0.08);
    --serif: 'Charter', 'Iowan Old Style', 'Georgia', 'Palatino', serif;
    --sans:  -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter',
             'Helvetica Neue', 'PingFang TC', 'Noto Sans TC', sans-serif;
}

/* 全域字體 */
html, body, [class*="css"], .stApp {
    font-family: var(--sans) !important;
    color: var(--text-primary);
}

/* 主容器 */
.main .block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1240px;
}

.stApp {
    background: var(--bg-primary) !important;
}

/* ── 標題層級：襯線字 + 暖色 ── */
h1 {
    font-family: var(--serif) !important;
    font-weight: 600 !important;
    font-size: 2.1rem !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.4px;
    line-height: 1.25;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
    margin-bottom: 20px !important;
}
h2 {
    font-family: var(--serif) !important;
    font-weight: 600 !important;
    font-size: 1.5rem !important;
    color: var(--text-primary) !important;
    margin-top: 1.8rem !important;
    letter-spacing: -0.2px;
}
h3 {
    font-family: var(--serif) !important;
    font-weight: 600 !important;
    font-size: 1.2rem !important;
    color: var(--accent) !important;
    margin-top: 1.4rem !important;
}
h4, h5, h6 {
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

/* 一般段落 */
p, li {
    color: var(--text-primary);
    line-height: 1.6;
}
strong, b { color: var(--text-primary); font-weight: 600; }

/* ── 分隔線：極簡 ── */
hr {
    border: none !important;
    height: 1px !important;
    background: var(--border) !important;
    margin: 2rem 0 !important;
}

/* ── 自訂指標卡 .metric-card ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    margin: 8px 0;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.metric-card:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--border-strong);
}
.metric-card h2 {
    margin: 0 !important;
    border: none !important;
    padding: 0 !important;
    font-size: 2rem !important;
    color: var(--accent) !important;
    font-family: var(--serif) !important;
}
.metric-card p {
    margin: 4px 0 0;
    color: var(--text-secondary);
    font-size: 0.85rem;
}

/* ── 小組卡片 ── */
.group-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 16px;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s;
}
.group-card:hover { box-shadow: var(--shadow-md); }
.group-header {
    background: var(--accent);
    padding: 10px 16px;
    font-size: 0.92rem;
    font-weight: 600;
    color: #fff;
    letter-spacing: 0.3px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.team-row {
    display: flex;
    align-items: center;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text-primary);
    font-size: 0.9rem;
    transition: background 0.15s;
}
.team-row:last-child { border-bottom: none; }
.team-row:hover { background: var(--bg-elevated); }
.team-flag { font-size: 1.3rem; margin-right: 10px; flex-shrink: 0; }
.team-cn { font-weight: 600; color: var(--text-primary); margin-right: 6px; min-width: 60px; }
.team-en { color: var(--text-secondary); flex: 1; font-size: 0.82rem; }
.team-rank {
    font-size: 0.72rem;
    color: var(--accent);
    background: var(--accent-soft);
    border: 1px solid var(--border);
    padding: 2px 8px;
    border-radius: 999px;
    flex-shrink: 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    padding: 0;
    border-radius: 0;
    border: none;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px 6px 0 0;
    padding: 8px 16px;
    font-weight: 500;
    color: var(--text-secondary);
    transition: color 0.15s, background 0.15s;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
}
.stTabs [data-baseweb="tab"]:hover {
    background: var(--bg-elevated);
    color: var(--text-primary);
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    font-weight: 600 !important;
    box-shadow: none;
}

/* ── Metric ── */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: var(--shadow-sm);
}
div[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
}
div[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-weight: 600 !important;
    font-family: var(--serif) !important;
}
div[data-testid="stMetricDelta"] {
    color: var(--text-secondary) !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    background: var(--bg-card) !important;
    box-shadow: var(--shadow-sm);
    margin-bottom: 8px;
}
div[data-testid="stExpander"] summary {
    font-weight: 500 !important;
    color: var(--text-primary) !important;
    padding: 12px 18px !important;
    background: transparent !important;
    border-left: none !important;
}
div[data-testid="stExpander"] summary:hover {
    background: var(--bg-elevated) !important;
}

/* ── Alerts / Info ── */
div[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-weight: 400;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
}
div[data-testid="stAlert"] p {
    color: var(--text-primary) !important;
}

/* ── 球隊對比表（保留白底結構） ── */
.cmp-grid {
    display: grid;
    grid-template-columns: 1.4fr 1fr 1fr;
    margin-top: 10px;
    background: var(--bg-card);
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
}
.cmp-head {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    font-weight: 600;
    padding: 12px 14px;
    border-bottom: 1px solid var(--border);
}
.cmp-head-c { text-align: center; }
.cmp-head-red { border-bottom: 2px solid var(--accent) !important; }
.cmp-head-blue { border-bottom: 2px solid var(--accent-2) !important; }
.cmp-label {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-weight: 500;
    font-size: 0.95rem;
    padding: 14px;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
}
.cmp-cell {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-weight: 600;
    text-align: center;
    padding: 14px;
    font-size: 1.05rem;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
}
.cmp-dim {
    background: var(--bg-card) !important;
    color: var(--text-tertiary) !important;
    font-weight: 400;
    text-align: center;
    padding: 14px;
    font-size: 1.05rem;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
}
.cmp-win {
    background: var(--accent-soft) !important;
    color: var(--accent-hover) !important;
    font-weight: 700 !important;
    text-align: center;
    padding: 14px;
    font-size: 1.15rem;
    border-top: 2px solid var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
.cmp-dir {
    color: var(--accent);
    font-weight: 600;
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
.cmp-chip-red { background: var(--accent); }
.cmp-chip-blue { background: var(--accent-2); }

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

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-elevated) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary);
}
section[data-testid="stSidebar"] .stRadio label {
    padding: 6px 8px;
    border-radius: 6px;
    transition: background 0.15s;
    color: var(--text-primary);
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: var(--bg-card);
}
section[data-testid="stSidebar"] hr {
    margin: 1rem 0 !important;
    background: var(--border) !important;
}

/* ── Selectbox / Multiselect ── */
div[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
}
div[data-baseweb="select"] * {
    color: var(--text-primary) !important;
}

/* ── Button ── */
.stButton button {
    background: var(--accent);
    color: #fff;
    border: 1px solid var(--accent);
    border-radius: 8px;
    font-weight: 500;
    padding: 8px 18px;
    transition: background 0.15s, transform 0.15s;
    box-shadow: var(--shadow-sm);
}
.stButton button:hover {
    background: var(--accent-hover);
    border-color: var(--accent-hover);
    color: #fff;
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

/* ── Caption ── */
[data-testid="stCaptionContainer"] {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    line-height: 1.6;
}

/* ── Plotly 圖表外框（極簡） ── */
.js-plotly-plot {
    border-radius: 8px;
    background: var(--bg-card);
    padding: 8px;
    border: 1px solid var(--border);
}

/* ── Progress bar ── */
div[data-testid="stProgress"] > div > div {
    background: var(--bg-subtle);
    border-radius: 4px;
}
div[data-testid="stProgressBar"] > div > div {
    background: var(--accent) !important;
}

/* ── Markdown links ── */
a {
    color: var(--accent);
    text-decoration: none;
}
a:hover {
    color: var(--accent-hover);
    text-decoration: underline;
}

/* ── 響應式：手機版 ── */
@media (max-width: 768px) {
    .main .block-container { padding: 1rem 0.6rem; }
    h1 { font-size: 1.6rem !important; }
    h2 { font-size: 1.25rem !important; }
    h3 { font-size: 1.05rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# TEAM_INFO 已移至 constants.py（單一資料來源），於檔案頂部 import
# ============================================================


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

# WC_2026_GROUPS 已移至 constants.py（單一資料來源），於檔案頂部 import

# 小組賽完整賽程（台灣時間 UTC+8）— FIFA 2026 官方賽程，全 72 場
# 每筆：(日期 "M/D", 開球時間 "H:MM", 組別, 主隊, 客隊)；隊名為 constants.WC_2026_GROUPS 內部鍵值
# 已依台灣時間先後排序；此為唯一資料來源，「完整賽程」頁與「預測」頁開球時間皆由此推導
WC_2026_GROUP_FIXTURES = [
    # ── 第 1 輪 (MD1) 6/12–6/18 ──
    ("6/12", "3:00",  "A", "Mexico", "South Africa"),
    ("6/12", "10:00", "A", "South Korea", "Czechia"),
    ("6/13", "3:00",  "B", "Canada", "Bosnia and Herzegovina"),
    ("6/13", "9:00",  "D", "USA", "Paraguay"),
    ("6/14", "3:00",  "B", "Qatar", "Switzerland"),
    ("6/14", "6:00",  "C", "Brazil", "Morocco"),
    ("6/14", "9:00",  "C", "Haiti", "Scotland"),
    ("6/14", "12:00", "D", "Australia", "Turkiye"),
    ("6/15", "1:00",  "E", "Germany", "Curacao"),
    ("6/15", "4:00",  "F", "Netherlands", "Japan"),
    ("6/15", "7:00",  "E", "Ivory Coast", "Ecuador"),
    ("6/15", "10:00", "F", "Sweden", "Tunisia"),
    ("6/16", "0:00",  "H", "Spain", "Cape Verde"),
    ("6/16", "3:00",  "G", "Belgium", "Egypt"),
    ("6/16", "6:00",  "H", "Saudi Arabia", "Uruguay"),
    ("6/16", "9:00",  "G", "Iran", "New Zealand"),
    ("6/17", "3:00",  "I", "France", "Senegal"),
    ("6/17", "6:00",  "I", "Iraq", "Norway"),
    ("6/17", "9:00",  "J", "Argentina", "Algeria"),
    ("6/17", "12:00", "J", "Austria", "Jordan"),
    ("6/18", "1:00",  "K", "Portugal", "DR Congo"),
    ("6/18", "4:00",  "L", "England", "Croatia"),
    ("6/18", "7:00",  "L", "Ghana", "Panama"),
    ("6/18", "10:00", "K", "Uzbekistan", "Colombia"),
    # ── 第 2 輪 (MD2) 6/19–6/24 ──
    ("6/19", "0:00",  "A", "Czechia", "South Africa"),
    ("6/19", "3:00",  "B", "Switzerland", "Bosnia and Herzegovina"),
    ("6/19", "6:00",  "B", "Canada", "Qatar"),
    ("6/19", "9:00",  "A", "Mexico", "South Korea"),
    ("6/20", "3:00",  "D", "USA", "Australia"),
    ("6/20", "6:00",  "C", "Scotland", "Morocco"),
    ("6/20", "8:30",  "C", "Brazil", "Haiti"),
    ("6/20", "11:00", "D", "Turkiye", "Paraguay"),
    ("6/21", "1:00",  "F", "Netherlands", "Sweden"),
    ("6/21", "4:00",  "E", "Germany", "Ivory Coast"),
    ("6/21", "8:00",  "E", "Ecuador", "Curacao"),
    ("6/21", "12:00", "F", "Tunisia", "Japan"),
    ("6/22", "0:00",  "H", "Spain", "Saudi Arabia"),
    ("6/22", "3:00",  "G", "Belgium", "Iran"),
    ("6/22", "6:00",  "H", "Uruguay", "Cape Verde"),
    ("6/22", "9:00",  "G", "New Zealand", "Egypt"),
    ("6/23", "1:00",  "J", "Argentina", "Austria"),
    ("6/23", "5:00",  "I", "France", "Iraq"),
    ("6/23", "8:00",  "I", "Norway", "Senegal"),
    ("6/23", "11:00", "J", "Jordan", "Algeria"),
    ("6/24", "1:00",  "K", "Portugal", "Uzbekistan"),
    ("6/24", "4:00",  "L", "England", "Ghana"),
    ("6/24", "7:00",  "L", "Panama", "Croatia"),
    ("6/24", "10:00", "K", "Colombia", "DR Congo"),
    # ── 第 3 輪 (MD3，同組同時開踢) 6/25–6/28 ──
    ("6/25", "3:00",  "B", "Switzerland", "Canada"),
    ("6/25", "3:00",  "B", "Bosnia and Herzegovina", "Qatar"),
    ("6/25", "6:00",  "C", "Scotland", "Brazil"),
    ("6/25", "6:00",  "C", "Morocco", "Haiti"),
    ("6/25", "9:00",  "A", "Czechia", "Mexico"),
    ("6/25", "9:00",  "A", "South Africa", "South Korea"),
    ("6/26", "4:00",  "E", "Ecuador", "Germany"),
    ("6/26", "4:00",  "E", "Curacao", "Ivory Coast"),
    ("6/26", "7:00",  "F", "Japan", "Sweden"),
    ("6/26", "7:00",  "F", "Tunisia", "Netherlands"),
    ("6/26", "10:00", "D", "Turkiye", "USA"),
    ("6/26", "10:00", "D", "Paraguay", "Australia"),
    ("6/27", "3:00",  "I", "Norway", "France"),
    ("6/27", "3:00",  "I", "Senegal", "Iraq"),
    ("6/27", "8:00",  "H", "Cape Verde", "Saudi Arabia"),
    ("6/27", "8:00",  "H", "Uruguay", "Spain"),
    ("6/27", "11:00", "G", "Egypt", "Iran"),
    ("6/27", "11:00", "G", "New Zealand", "Belgium"),
    ("6/28", "5:00",  "L", "Panama", "England"),
    ("6/28", "5:00",  "L", "Croatia", "Ghana"),
    ("6/28", "7:30",  "K", "Colombia", "Portugal"),
    ("6/28", "7:30",  "K", "DR Congo", "Uzbekistan"),
    ("6/28", "10:00", "J", "Algeria", "Austria"),
    ("6/28", "10:00", "J", "Jordan", "Argentina"),
]

# 星期對照（台灣時間）
WC_2026_WEEKDAY = {
    "6/12": "五", "6/13": "六", "6/14": "日", "6/15": "一", "6/16": "二",
    "6/17": "三", "6/18": "四", "6/19": "五", "6/20": "六", "6/21": "日",
    "6/22": "一", "6/23": "二", "6/24": "三", "6/25": "四", "6/26": "五",
    "6/27": "六", "6/28": "日",
}

# 由賽程表推導 (組, 隊index i, 隊index j) -> "M/D H:MM"，供「2026 預測」頁查詢開球時間
def _build_group_schedule() -> dict:
    s = {}
    for date, t, g, home, away in WC_2026_GROUP_FIXTURES:
        roster = WC_2026_GROUPS.get(g, [])
        try:
            i, j = sorted((roster.index(home), roster.index(away)))
        except ValueError:
            continue  # 隊名與分組不符時略過，避免整頁崩潰
        s[(g, i, j)] = f"{date} {t}"
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
    """該球隊歷史世界盃正賽場次（大賽經驗值）。

    註：資料集 tournament 欄無小組賽/淘汰賽之分，故此處計算的是
    「世界盃正賽（排除資格賽）總場次」，作為球隊的大賽經驗代理指標，
    而非僅淘汰賽。場次越多代表越常打進並深入世界盃。
    """
    wc_matches = df[
        (df['tournament'].str.contains('World Cup', na=False)) &
        (~df['tournament'].str.contains('qualif', case=False, na=False)) &
        ((df['home_team'] == team) | (df['away_team'] == team))
    ]
    return len(wc_matches)

# CONFED_BONUS 與 CONFED_MAP 已移至 constants.py（單一資料來源），於檔案頂部 import

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
    """取出某屆世界盃實際比賽，用歷史資料當特徵（用於 walk-forward 驗證）"""
    samples = []
    # 直接迭代該年實際 WC 比賽（排除資格賽），不再受限於 2026 分組
    # 這樣 2010/2014/2018 等屆所有比賽都能進入驗證集
    wc = match_df[
        (match_df['year'] == year) &
        (match_df['tournament'].str.contains('World Cup', na=False)) &
        (~match_df['tournament'].str.contains('qualif', case=False, na=False))
    ]
    for _, row in wc.iterrows():
        t1, t2 = row['home_team'], row['away_team']
        try:
            feat = create_features_v2(t1, t2, year, match_df, fifa_df)
            gf, ga = row['home_score'], row['away_score']
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
    # v2: CONMEBOL 1.00→0.97（與 UEFA 0.96 拉近，避免南美 λ 被過度放大）
    CONFED_SCALE = {'UEFA': 0.96, 'CONMEBOL': 0.97, 'CONCACAF': 0.88,
                    'AFC': 0.84, 'CAF': 0.78, 'OFC': 0.72}
    cs1 = CONFED_SCALE.get(CONFED_MAP.get(team1, 'CAF'), 0.84)
    cs2 = CONFED_SCALE.get(CONFED_MAP.get(team2, 'CAF'), 0.84)

    # Dixon-Coles 標準形式：λ_A = (atk_A / μ) × (vul_B / μ) × μ = atk_A × vul_B / μ
    #   atk: 進攻強度（高=強）
    #   vul: 防守脆弱度（=avg_conceded，高=容易失球）
    # 弱聯盟修正：
    #   - 進攻數據被高估（對手弱）→ atk *= cs（弱聯盟 cs 小，下調 atk）
    #   - 失球數據被低估（對手弱）→ vul /= sqrt(cs)（弱聯盟實際更易失球，上調 vul）
    atk1 = min(2.5, s1['avg_goals'] * cs1)
    atk2 = min(2.5, s2['avg_goals'] * cs2)
    vul1 = min(2.5, s1['avg_conceded'] / max(0.5, cs1 ** 0.5))
    vul2 = min(2.5, s2['avg_conceded'] / max(0.5, cs2 ** 0.5))

    lam1_dc = min(2.5, max(0.3, atk1 * vul2 / LEAGUE_AVG))
    lam2_dc = min(2.5, max(0.3, atk2 * vul1 / LEAGUE_AVG))

    # ── λ 多因素融合：DC 基礎 × FIFA 排名 × 主將 × 近期狀態 × 淘汰賽經驗 ──
    pts1, pts2 = team_pts(team1), team_pts(team2)
    # 防呆：分母最小 1 避免 ZeroDivision；pts 經驗範圍 0-2000 + 300 緩衝
    rank_factor = ((max(pts1, 0) + 300) / max(pts2 + 300, 1)) ** 0.22

    ovr1, ovr2 = squad_ovr(team1), squad_ovr(team2)
    squad_factor = (max(ovr1, 1.0) / max(ovr2, 1.0)) ** 0.35

    form1 = compute_recent_form(match_df, team1, year)
    form2 = compute_recent_form(match_df, team2, year)
    form_factor = ((form1 + 0.3) / max(form2 + 0.3, 0.1)) ** 0.18

    exp1 = compute_knockout_exp(match_df, team1)
    exp2 = compute_knockout_exp(match_df, team2)
    exp_factor = ((exp1 + 10) / max(exp2 + 10, 1)) ** 0.08

    # 浮點下溢保護：composite_factor 限制在合理區間 [0.5, 2.0]
    # 即使所有因子同向極端疊加也不會讓單側 λ 被打到極值
    composite_factor = max(0.5, min(2.0,
                                     rank_factor * squad_factor * form_factor * exp_factor))
    lam1 = min(2.8, max(0.3, lam1_dc * composite_factor))
    lam2 = min(2.8, max(0.3, lam2_dc / composite_factor))

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

    # ── 第三層：加權融合（XGBoost 60% + Poisson 40%）──
    # XGBoost clf 是 Walk-Forward 實際驗證(52%)的模型，主導勝負「方向」；
    # Poisson 由 λ（攻防強度）推導，輔助比分形狀。
    # v3.1：clf 權重 0.2→0.6——2026 實戰顯示原 0.2/0.8 讓 Poisson composite
    #       偶爾蓋過 clf 正確判斷（如海地-蘇格蘭判反）；提高 clf 權重對齊已驗證模型。
    W_CLF, W_POI = 0.60, 0.40
    prob_win  = W_CLF * clf_win  + W_POI * poi_win
    prob_draw = W_CLF * clf_draw + W_POI * poi_draw
    prob_loss = W_CLF * clf_loss + W_POI * poi_loss
    _t3 = prob_win + prob_draw + prob_loss
    prob_win /= _t3; prob_draw /= _t3; prob_loss /= _t3

    # ── MAP 比分：在「融合勝負方向」內找最高 Poisson 機率 cell ──
    # 為什麼要約束方向：當 λ 接近時，最高的單格機率常落在 (1,1) 等平局
    #   即使整體勝率(40%)大於平局(27%)。約束後 ★ 標的是「符合主預測方向的
    #   最強比分」，與顯示的機率方向一致，避免「20%平局卻每場都平」的悖論
    outcome = ('win' if prob_win >= prob_draw and prob_win >= prob_loss
               else 'draw' if prob_draw >= prob_loss else 'loss')

    # 高和局率規則（Hermes 校準：T=0.28、margin=0.05、AND 邏輯）：
    # 當「和局率夠高」且「與最高勝率夠接近」(真．平局訊號) 時，直接預測和局。
    # ⚠️ 經 16 場實戰回測：今年和局多為「熱門掉分」型(模型 P_draw 僅 0.18–0.32、
    #    仍偏向贏家)，此門檻目前幾乎不觸發、也不傷決勝命中；主要為日後真正勢均
    #    力敵的對戰武裝。要真正多預測和局需做類別權重校準(會犧牲部分總準確率)。
    _DRAW_T, _DRAW_MARGIN = 0.28, 0.05
    if prob_draw >= _DRAW_T and (max(prob_win, prob_loss) - prob_draw) <= _DRAW_MARGIN:
        outcome = 'draw'

    # 規範化到字母序，確保同場比賽比分不因呼叫順序而改變
    t_can1, t_can2 = sorted([team1, team2])
    is_canonical = (team1 == t_can1)
    lam_c1 = lam1 if is_canonical else lam2
    lam_c2 = lam2 if is_canonical else lam1
    out_c = outcome if is_canonical else (
        'loss' if outcome == 'win' else ('win' if outcome == 'loss' else 'draw'))

    best_prob, gc1, gc2 = -1.0, 0, 0
    for g1 in range(8):
        for g2 in range(8):
            so = 'win' if g1 > g2 else ('draw' if g1 == g2 else 'loss')
            if so != out_c:
                continue
            p = poisson.pmf(g1, lam_c1) * poisson.pmf(g2, lam_c2)
            if p > best_prob:
                best_prob, gc1, gc2 = p, g1, g2

    goal1 = gc1 if is_canonical else gc2
    goal2 = gc2 if is_canonical else gc1
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
# 小組賽：模型預測比分 ＋ ESPN 即時實際比分
# ============================================================
@st.cache_data(show_spinner="計算 72 場小組賽預測比分中…")
def get_group_stage_predictions() -> dict:
    """回傳 {(home_en, away_en): {'g1','g2','win','draw','loss'}}；模型不可用時回傳空 dict。"""
    md = load_match_data()
    fd = load_fifa_ranking()
    out: dict = {}
    pre = load_pretrained()
    if pre is not None:
        clf, p1, p2, fc = pre['clf'], pre['poisson1'], pre['poisson2'], pre['feat_cols']
    else:
        # 線上若無 pkl 或反序列化失敗，與「2026 預測」頁一致 → 即時訓練（結果有快取）
        res = train_models_walkforward(md, fd)
        if not res or res[0] is None:
            return out
        clf, p1, p2, fc = res[0], res[3], res[4], res[5]
    for _d, _t, _g, home, away in WC_2026_GROUP_FIXTURES:
        try:
            pr = predict_match(home, away, 2026, md, fd, clf, p1, p2, fc)
            out[(home, away)] = {'g1': pr['goal1'], 'g2': pr['goal2'],
                                 'win': pr['win_prob'], 'draw': pr['draw_prob'], 'loss': pr['loss_prob']}
        except Exception:
            out[(home, away)] = None
    return out


# ESPN 公開 API 隊名 → 本 app 內部鍵值（僅列與內部不同者）
_ESPN_NAME_MAP = {
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Congo DR': 'DR Congo',
    'Curaçao': 'Curacao',
    'Türkiye': 'Turkiye',
    'United States': 'USA',
}


@st.cache_data(ttl=30, show_spinner=False)
def fetch_wc_live_scores() -> dict:
    """從 ESPN 公開 API 抓 2026 世界盃即時/實際比分。
    回傳 {frozenset({home_en, away_en}): {'state','detail','scores':{team:int|None}}}；失敗回傳空 dict。"""
    import urllib.request
    import json as _json
    url = ('https://site.api.espn.com/apis/site/v2/sports/soccer/'
           'fifa.world/scoreboard?dates=20260611-20260629')
    out: dict = {}
    try:
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = _json.load(op.open(req, timeout=8))
    except Exception:
        return out
    for e in data.get('events', []):
        comp = (e.get('competitions') or [{}])[0]
        cs = comp.get('competitors', [])
        if len(cs) != 2:
            continue
        teams = []
        scores = {}
        for c in cs:
            raw = (c.get('team') or {}).get('displayName', '')
            nm = _ESPN_NAME_MAP.get(raw, raw)
            teams.append(nm)
            try:
                scores[nm] = int(c.get('score'))
            except (TypeError, ValueError):
                scores[nm] = None
        stt = ((e.get('status') or {}).get('type') or {})
        out[frozenset(teams)] = {
            'state': stt.get('state', 'pre'),
            'detail': stt.get('shortDetail') or stt.get('description') or '',
            'scores': scores,
        }
    return out


def render_group_stage_schedule() -> None:
    """渲染小組賽賽程（依日期）＋ 模型預測比分 ＋ ESPN 即時實際比分。"""
    from itertools import groupby as _groupby
    preds = get_group_stage_predictions()
    live = fetch_wc_live_scores()
    grp_color = {
        'A': '#e94560', 'B': '#f7943e', 'C': '#f7c948', 'D': '#7bd88f',
        'E': '#3fb6b2', 'F': '#4aa3ff', 'G': '#6c8cff', 'H': '#a06cff',
        'I': '#e066c4', 'J': '#ff6b9d', 'K': '#c0a35e', 'L': '#5ec6c0',
    }
    # 賽果統計（完賽場次）：勝負命中、平局命中、進球偏差
    n_done = hit = draw_act = draw_hit = 0
    pred_goal_sum = act_goal_sum = 0
    day_blocks = ""
    for date, items in _groupby(WC_2026_GROUP_FIXTURES, key=lambda m: m[0]):
        rows = ""
        for (_d, _t, g, home, away) in items:
            ih = TEAM_INFO.get(home, {}); ia = TEAM_INFO.get(away, {})
            gc = grp_color.get(g, '#4a7ea8')
            # 主場放右邊：左＝客隊(away)、右＝主隊(home)
            iso_L = ia.get('iso', 'un'); cn_L = ia.get('cn', away)
            iso_R = ih.get('iso', 'un'); cn_R = ih.get('cn', home)

            # 〔預〕模型最可能比分（客-主）＋ 預測勝方勝率
            pr = preds.get((home, away))
            pred_dir = None  # 1=主勝, -1=客勝, 0=平（用於完賽後比對命中）
            if pr:
                gh, ga = pr['g1'], pr['g2']            # gh=主隊, ga=客隊
                if gh > ga:                             # 主勝 → 右側（主隊）亮
                    cL, cR = '#7fa6b5', '#00d4ff'; pred_dir = 1; pick_prob = pr.get('win', 0.0)
                elif gh < ga:                           # 客勝 → 左側（客隊）亮
                    cL, cR = '#00d4ff', '#7fa6b5'; pred_dir = -1; pick_prob = pr.get('loss', 0.0)
                else:
                    cL = cR = '#cbb46b'; pred_dir = 0; pick_prob = pr.get('draw', 0.0)
                pred_html = (f'<span style="color:#5b7a8a;font-size:0.6rem;margin-right:2px;">預</span>'
                             f'<span style="color:{cL};font-weight:800;">{ga}</span>'
                             f'<span style="color:#5b7a8a;">-</span>'
                             f'<span style="color:{cR};font-weight:800;">{gh}</span>'
                             f'<span style="color:#8aa0ae;font-size:0.6rem;margin-left:5px;">勝{pick_prob:.0%}</span>'
                             f'<span style="color:#cbb46b;font-size:0.6rem;margin-left:3px;">和{pr.get("draw", 0.0):.0%}</span>')
            else:
                pred_html = '<span style="color:#5b7a8a;font-size:0.62rem;">預 –</span>'

            # 〔實〕ESPN 實際/即時比分（客-主）＋ 完賽後「勝負命中」標記
            lv = live.get(frozenset((home, away)))
            if lv and lv.get('state') in ('in', 'post'):
                sh = lv['scores'].get(home); sa = lv['scores'].get(away)
                s_home = '?' if sh is None else sh
                s_away = '?' if sa is None else sa
                if lv['state'] == 'in':
                    act_html = (f'<span style="color:#ff4d4f;font-weight:800;">● {s_away}-{s_home}</span>'
                                f'<span style="color:#ff8f8f;font-size:0.58rem;margin-left:3px;">LIVE</span>')
                else:
                    hit_html = ''
                    if pred_dir is not None and isinstance(sh, int) and isinstance(sa, int):
                        act_dir = 1 if sh > sa else (-1 if sh < sa else 0)
                        n_done += 1
                        hit += int(act_dir == pred_dir)
                        if act_dir == 0:
                            draw_act += 1
                            draw_hit += int(pred_dir == 0)
                        pred_goal_sum += gh + ga
                        act_goal_sum += sh + sa
                        if act_dir == pred_dir:
                            hit_html = ('<span style="color:#36c275;font-weight:800;'
                                        'font-size:0.62rem;margin-left:5px;">✓命中</span>')
                        else:
                            hit_html = ('<span style="color:#9aa0a6;font-weight:800;'
                                        'font-size:0.62rem;margin-left:5px;">✗</span>')
                    act_html = (f'<span style="color:#5b7a8a;font-size:0.6rem;margin-right:2px;">實</span>'
                                f'<span style="color:#f7c948;font-weight:800;">{s_away}-{s_home}</span>'
                                f'{hit_html}')
            else:
                act_html = '<span style="color:#46566a;font-size:0.64rem;">未開賽</span>'

            rows += (
                f'<div style="display:flex;align-items:center;gap:8px;padding:6px 8px;'
                f'border-bottom:1px solid rgba(100,150,200,0.12);">'
                f'<span style="color:#00d4ff;font-weight:700;min-width:50px;'
                f'font-variant-numeric:tabular-nums;font-size:0.9rem;">{_t}</span>'
                f'<span style="background:{gc};color:#0b1220;font-weight:800;border-radius:4px;'
                f'padding:1px 8px;font-size:0.78rem;min-width:24px;text-align:center;">{g}</span>'
                f'<span style="flex:1;text-align:right;color:#e8eef6;font-weight:600;font-size:0.93rem;">{cn_L}</span>'
                f'<img src="https://flagcdn.com/40x30/{iso_L}.png" style="height:18px;border-radius:2px;">'
                f'<span style="display:inline-flex;flex-direction:column;align-items:center;'
                f'min-width:188px;line-height:1.25;gap:1px;font-variant-numeric:tabular-nums;">'
                f'<span style="font-size:0.95rem;letter-spacing:0.5px;">{pred_html}</span>'
                f'<span style="font-size:0.8rem;">{act_html}</span>'
                f'</span>'
                f'<img src="https://flagcdn.com/40x30/{iso_R}.png" style="height:18px;border-radius:2px;">'
                f'<span style="flex:1;text-align:left;color:#e8eef6;font-weight:600;font-size:0.93rem;">{cn_R}'
                f'<span style="color:#f7943e;font-size:0.6rem;margin-left:3px;vertical-align:super;">主</span>'
                f'</span>'
                f'</div>'
            )
        wd = WC_2026_WEEKDAY.get(date, "")
        day_blocks += (
            f'<div style="margin:10px 0 0;padding:5px 12px;background:rgba(73,124,160,0.20);'
            f'border-left:3px solid #4a7ea8;border-radius:5px;color:#cfe3f5;font-weight:800;'
            f'font-size:0.95rem;">{date}（{wd}）</div>{rows}'
        )
    if n_done > 0:
        acc = hit / n_done
        bias = (act_goal_sum - pred_goal_sum) / n_done
        summary_html = (
            f'<div style="margin:0 0 10px;padding:10px 14px;background:rgba(54,194,117,0.10);'
            f'border:1px solid rgba(54,194,117,0.35);border-radius:8px;line-height:1.8;">'
            f'<span style="color:#cfe3f5;font-weight:800;font-size:0.95rem;">📊 賽果校準（已完賽 {n_done} 場）</span>'
            f'<span style="color:#36c275;font-weight:800;margin-left:12px;">勝負命中 {hit}/{n_done}（{acc:.0%}）</span>'
            f'<span style="color:#f7c948;margin-left:12px;font-size:0.88rem;">平局 {draw_act} 場 · 命中 {draw_hit}</span>'
            f'<span style="color:#8aa0ae;margin-left:12px;font-size:0.88rem;">進球偏差 {"+" if bias >= 0 else ""}{bias:.1f} 球/場（實際−預測）</span>'
            f'</div>'
        )
    else:
        summary_html = ('<div style="margin:0 0 10px;color:#8aa0ae;font-size:0.85rem;">'
                        '📊 賽果校準：尚無完賽場次，開賽後自動統計勝負命中率與進球偏差。</div>')
    st.markdown(
        f'<div style="background:#091525;border-radius:12px;padding:8px 14px 14px;'
        f'font-family:\'Noto Sans TC\',sans-serif;">{summary_html}{day_blocks}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================
def monte_carlo(match_df, fifa_df, clf, poisson1, poisson2, feat_cols, n_sims=10000):
    """Monte Carlo 奪冠模擬 — 動態赔率 + XGBoost 淘汰賽"""
    np.random.seed(42)
    all_teams = [t for ts in WC_2026_GROUPS.values() for t in ts]
    win_count = {t: 0 for t in all_teams}
    champion_count = {t: 0 for t in all_teams}

    # UEFA/CONMEBOL 基礎勝率加成（v2: CONMEBOL 0.10→0.06，與 constants 一致）
    BASE_CONFED_BONUS = {
        'UEFA': 0.08, 'CONMEBOL': 0.06, 'CONCACAF': 0.02,
        'AFC': 0.01, 'CAF': 0.00, 'OFC': 0.00
    }

    # 主將陣容修正：把市場最看重的陣容納入 MC（與 predict_match 同邏輯）
    _SQUAD_POW = 0.35
    def _squad_odds_mult(t1, t2):
        return (squad_ovr(t1) / max(squad_ovr(t2), 1.0)) ** _SQUAD_POW

    def team_base_winrate(team):
        """球隊基礎勝率：FIFA排名轉換 + 足聯加成"""
        pts = team_pts(team)
        base = 1 / (1 + np.exp(-2.0 * (pts - 1500) / 800))
        confed = CONFED_MAP.get(team, 'CAF')
        return min(base * 0.85 + BASE_CONFED_BONUS.get(confed, 0), 0.92)

    def sim_group_match(t1, t2):
        """小組賽單場模擬：動態p_draw + 陣容修正，回傳 (t1積分, t2積分)"""
        bw1 = team_base_winrate(t1)
        bw2 = team_base_winrate(t2)
        rank_diff = abs(team_rank(t1) - team_rank(t2))
        p_draw = max(0.15, min(0.28, 0.28 - rank_diff * 0.005))
        ratio = bw1 / (bw1 + bw2)
        # 陣容修正：以勝負勝率比乘上 (ovr1/ovr2)^0.35
        sf = _squad_odds_mult(t1, t2)
        ratio = ratio * sf / (ratio * sf + (1 - ratio))
        p_win = ratio * (1 - p_draw)
        r = np.random.random()
        if r < p_win:
            return 3, 0   # t1勝
        elif r < p_win + p_draw:
            return 1, 1   # 平局
        else:
            return 0, 3   # t2勝

    def sim_ko_match(t1, t2):
        """淘汰賽單場模擬（XGBoost 機率 + 陣容修正，無平局）"""
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
        # 陣容修正：把勝率以 odds 形式乘上陣容比
        sf = _squad_odds_mult(t1, t2)
        p_t1_wins = p_t1_wins * sf / (p_t1_wins * sf + (1 - p_t1_wins)) if 0 < p_t1_wins < 1 else p_t1_wins
        return t1 if np.random.random() < p_t1_wins else t2

    for _ in range(n_sims):
        # ── 小組賽：取各組前2晉級 + 記錄各組第3名 ──
        # 平局打破機制：用 random tiebreak（模擬 FIFA 進球差/淨進球的隨機性）
        sim_qualifiers = []
        third_place_info = []  # (pts, random_tiebreak, team) 各組第3名
        for g, teams in WC_2026_GROUPS.items():
            pts = {t: 0 for t in teams}
            for i, t1 in enumerate(teams):
                for t2 in teams[i+1:]:
                    p1, p2 = sim_group_match(t1, t2)
                    pts[t1] += p1; pts[t2] += p2
            # 平局時用隨機數打破（避免字母序的系統性偏差）
            sorted_pts = sorted(
                pts.items(),
                key=lambda x: (x[1], np.random.random()),
                reverse=True,
            )
            qualifiers = [t for t, _ in sorted_pts[:2]]
            sim_qualifiers.extend(qualifiers)
            win_count[qualifiers[0]] += 1
            third_team, third_pts = sorted_pts[2]
            third_place_info.append((third_pts, np.random.random(), third_team))

        # 2026 世界盃：24支小組前2 + 8支最佳第3名（按積分+隨機 tiebreak 排序）= 32強
        best_thirds = sorted(third_place_info, reverse=True)[:8]
        bracket = sim_qualifiers + [t for _, _, t in best_thirds]
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
# LOGO + 標題：固定（sticky）於側邊欄頂端，捲動時不跟著移動
_logo_path = os.path.join(os.path.dirname(__file__), 'assets', 'logo.png')
_logo_b64 = ''
if os.path.exists(_logo_path):
    with open(_logo_path, 'rb') as _lf:
        _logo_b64 = base64.b64encode(_lf.read()).decode()

# ── 版本標記：版本號＋日期＋實際部署 commit 短雜湊（線上可直接對照 GitHub）──
APP_VERSION = "v3.4"
APP_BUILD_DATE = "2026-06-16"
_app_build = f"{APP_VERSION} · {APP_BUILD_DATE}"

st.sidebar.markdown(
    f"""
    <style>
    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {{
        padding-top: 8px !important;
        padding-bottom: 16px !important;
    }}
    .wc-side-head {{
        padding: 4px 4px 10px;
    }}
    .wc-side-head img {{
        display: block; width: 72%; margin: 0 auto 8px;
        border-radius: 10px;
    }}
    .wc-side-title {{
        text-align: center;
        font-family: Charter, Georgia, serif; font-size: 1.2rem;
        font-weight: 600; color: #1f1e1c; letter-spacing: -0.2px;
    }}
    .wc-side-sub {{
        text-align: center;
        font-size: 0.76rem; color: #6b6760; margin-top: 2px;
    }}
    </style>
    <div class="wc-side-head">
        {'<img src="data:image/png;base64,' + _logo_b64 + '" alt="logo">' if _logo_b64 else ''}
        <div class="wc-side-title">World Cup 2026</div>
        <div class="wc-side-sub">ML 勝率分析 · {_app_build}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 背景主題曲已移除（不再自動播放音樂）──

# ── 分割版面：LOGO 固定於上，導航 + 資訊放進「固定高度可捲動容器」──
# st.container(height=N, key=...) 是原生捲動容器；用 key 對應的穩定 class（st-key-…）
# 把它的高度改為「填滿 LOGO 下方剩餘空間」，使整個側邊欄不外捲、LOGO 永遠釘在頂端。
st.sidebar.markdown(
    """
    <style>
    /* 導航容器：填滿 LOGO 下方到視窗底（內部捲動，外層側邊欄不捲）。
       同時鎖定容器本體＋其外層包裝層，否則包裝層仍保留固定 600px
       導致整個側邊欄外捲、LOGO 被一起捲走。
       為相容不同 Streamlit 版本（DOM 包裝層不同），三種選擇器都寫上，
       各版本只有對應者命中、其餘為無害的 no-op： */
    /* (a) 容器根元素：所有版本都有 st-key class
       314（header+LOGO+內距）；主題曲播放器已移除，不再額外扣高度 */
    section[data-testid="stSidebar"] .st-key-wc_navbox {
        height: calc(100vh - 314px) !important;
        max-height: calc(100vh - 314px) !important;
    }
    /* (b) 新版（1.5x）：st-key 即捲動層，外層 stLayoutWrapper 也要縮 */
    section[data-testid="stSidebar"] div[data-testid="stLayoutWrapper"]:has(> .st-key-wc_navbox) {
        height: calc(100vh - 314px) !important;
        max-height: calc(100vh - 314px) !important;
    }
    /* (c) 舊版：st-key 為外框，內層 stVerticalBlock 才是捲動層 → 撐滿外框 */
    section[data-testid="stSidebar"] .st-key-wc_navbox > div[data-testid="stVerticalBlock"] {
        height: 100% !important;
        max-height: 100% !important;
    }
    /* 容器內捲軸細一點、不突兀 */
    section[data-testid="stSidebar"] .st-key-wc_navbox ::-webkit-scrollbar { width: 6px; }
    section[data-testid="stSidebar"] .st-key-wc_navbox ::-webkit-scrollbar-thumb {
        background: #d8d2c8; border-radius: 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
_nav_box = st.sidebar.container(height=600, border=False, key="wc_navbox")
with _nav_box:
    page = st.radio("選擇頁面", [
        "📊 專題總覽",
        "🔮 2026 預測",
        "📈 數據分析",
        "🌍 各國分析",
        "🎯 球隊風格分群",
        "🏅 奪冠預測",
        "📅 完整賽程",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size:0.78rem; color:#6b6760; line-height:1.7; padding:4px;'>
            <div style='color:#1f1e1c; font-weight:600; margin-bottom:4px;'>模型架構</div>
            <div>XGBoost 60% + Dixon-Coles 40%</div>
            <div>+ Squad OVR + Monte Carlo 10k</div>
            <div style='height:12px;'></div>
            <div style='color:#1f1e1c; font-weight:600; margin-bottom:4px;'>訓練資料</div>
            <div>49,257 場 · 1990–2025</div>
            <div>67,894 筆 FIFA 排名</div>
            <div style='height:12px;'></div>
            <div style='font-size:0.72rem; color:#9b958a;'>
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
    st.markdown("**真實資料：49,257 場國際比賽 · 67,894 筆 FIFA 排名**")
    st.markdown("---")

    # ── 主視覺圖（hero banner，全寬融入頁面） ──
    _hero_path = os.path.join(os.path.dirname(__file__), 'assets', 'hero.jpg')
    if os.path.exists(_hero_path):
        st.markdown(
            """
            <style>
            div[data-testid="stImage"] img {
                border-radius: 12px;
                box-shadow: 0 4px 16px rgba(31,30,28,0.08);
                width: 100%;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.image(_hero_path, use_container_width=True)
        st.markdown("")

    match_df = load_match_data()
    fifa_df = load_fifa_ranking()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h2>49,257</h2><p>歷史比賽場次</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>48</h2><p>參賽球隊</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2>235</h2><p>國家球隊數</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h2>12</h2><p>小組數</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📋 2026 世界盃小組分組（🇺🇸🇨🇦🇲🇽 主辦）")

    # 4 列 CSS Grid：Claude 風格小組卡片（白底 + 細邊框）
    GRID_CSS = """
    <style>
    .grp-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 14px 0 20px;
    }
    .grp-card {
        background: #ffffff;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e8e5dd;
        box-shadow: 0 1px 2px rgba(31,30,28,0.04);
        display: flex;
        flex-direction: column;
        height: 218px;
        transition: box-shadow 0.2s;
    }
    .grp-card:hover {
        box-shadow: 0 4px 16px rgba(31,30,28,0.08);
    }
    .grp-hdr {
        background: #c96442;
        padding: 9px 14px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: 0.3px;
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
        justify-content: flex-start;
        overflow: hidden;
    }
    .grp-row {
        display: flex;
        align-items: center;
        padding: 7px 14px;
        border-bottom: 1px solid #f4f1ea;
        min-height: 42px;
        transition: background 0.15s;
    }
    .grp-row:hover { background: #faf9f5; }
    .grp-row:last-child { border-bottom: none; }
    .grp-flag { width: 28px; height: 21px; margin-right: 8px; flex-shrink: 0; object-fit: cover; border-radius: 2px; vertical-align: middle; }
    .grp-nm {
        font-size: 0.82rem;
        font-weight: 600;
        color: #1f1e1c;
        line-height: 1.3;
        word-break: break-word;
        overflow-wrap: anywhere;
        flex: 1;
    }
    .grp-en {
        font-size: 0.70rem;
        color: #9b958a;
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
        color: #c96442;
        background: #f3e9e4;
        border: 1px solid #e8e5dd;
        padding: 1px 7px;
        border-radius: 999px;
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

    st.caption("四層預測管線：每層處理結果送入下一層（另有第 5 層市場校準於「🏅 奪冠預測」頁）")

    # ── 階層式樣式 ──
    st.markdown(
        """
        <style>
        .arch-layer { background:#fff;border:1px solid #e8e5dd;border-left:4px solid #c96442;
                      border-radius:8px;padding:16px 20px;margin:6px 0;
                      box-shadow:0 1px 2px rgba(31,30,28,0.04); }
        .arch-layer-header { display:flex;align-items:center;gap:14px;margin-bottom:10px;
                             padding-bottom:8px;border-bottom:1px solid #f3e9e4; }
        .arch-badge { display:inline-flex;align-items:center;justify-content:center;
                      width:40px;height:40px;background:#1a2b5c;color:#f9c846;
                      border-radius:50%;font-weight:800;font-size:13px;
                      font-family:Charter,Georgia,serif;flex-shrink:0; }
        .arch-title { font-size:17px;font-weight:700;color:#1f1e1c;margin:0; }
        .arch-subtitle { font-size:12px;color:#6b6760;margin-top:2px; }
        .arch-weight-chip { margin-left:auto;background:#c96442;color:#fff;
                            padding:5px 14px;border-radius:999px;font-size:12px;
                            font-weight:700;letter-spacing:0.5px; }
        .arch-content { display:grid;grid-template-columns:1fr 1fr;gap:20px; }
        .arch-block-title { font-size:13px;font-weight:700;color:#1a2b5c;margin-bottom:6px; }
        .arch-block-body { font-size:13px;color:#1f1e1c;line-height:1.7; }
        .arch-formula { background:#1a2b5c;color:#f9c846;padding:10px 14px;border-radius:6px;
                        font-family:Consolas,'Courier New',monospace;font-size:14px;
                        text-align:center;margin:8px 0;font-weight:700; }
        .arch-arrow { display:flex;justify-content:center;align-items:center;
                      margin:-4px 0;color:#c96442;font-size:24px;font-weight:700; }
        .arch-output { background:linear-gradient(135deg,#1a2b5c 0%,#2a3b6c 100%);
                       color:#fff;border-radius:10px;padding:20px 24px;margin-top:4px;
                       text-align:center;box-shadow:0 4px 16px rgba(26,43,92,0.2); }
        .arch-output-label { color:#f9c846;font-size:12px;font-weight:700;
                             letter-spacing:2px;margin-bottom:6px; }
        .arch-output-content { font-size:16px;color:#fff;line-height:1.7; }
        .arch-output-content code { background:rgba(249,200,70,0.2);color:#f9c846;
                                    padding:2px 8px;border-radius:4px;
                                    font-family:Consolas,'Courier New',monospace; }
        .arch-table { width:100%;border-collapse:collapse;font-size:12px;margin-top:4px; }
        .arch-table th { background:#f3e9e4;color:#1a2b5c;padding:6px 8px;
                         text-align:left;font-weight:700; }
        .arch-table td { padding:5px 8px;border-bottom:1px solid #f3e9e4; }
        .arch-table .pow { color:#c96442;font-weight:700;
                           font-family:Consolas,'Courier New',monospace; }
        .arch-tip { background:#fef1c9;border-left:3px solid #f9c846;padding:8px 12px;
                    margin-top:10px;border-radius:4px;font-size:12px;color:#1f1e1c; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Layer 1: XGBoost ──
    st.markdown(
        """
        <div class="arch-layer">
            <div class="arch-layer-header">
                <span class="arch-badge">L1</span>
                <div>
                    <div class="arch-title">🤖 XGBoost 分類器</div>
                    <div class="arch-subtitle">直接輸出 P(勝 / 平 / 負) 三向機率分布</div>
                </div>
                <span class="arch-weight-chip">外層權重 60%</span>
            </div>
            <div class="arch-content">
                <div>
                    <div class="arch-block-title">📥 輸入特徵</div>
                    <div class="arch-block-body">
                        • FIFA 積分差 (pts_diff)<br>
                        • 近 8 年勝率差 (win_rate_diff)<br>
                        • 場均進球差 (avg_goals_diff)<br>
                        • 足聯歸屬 (confed_bonus)
                    </div>
                </div>
                <div>
                    <div class="arch-block-title">⚙️ 訓練設定</div>
                    <div class="arch-block-body">
                        • 樣本：1990-2025 國際賽（49,257 場）<br>
                        • Walk-Forward 滾動驗證<br>
                        • 正向＋反向對稱平均<br>
                        &nbsp;&nbsp;&nbsp;→ 消除主場偏差
                    </div>
                </div>
            </div>
        </div>
        <div class="arch-arrow">↓</div>
        """,
        unsafe_allow_html=True,
    )

    # ── Layer 2: Dixon-Coles ──
    st.markdown(
        """
        <div class="arch-layer">
            <div class="arch-layer-header">
                <span class="arch-badge">L2</span>
                <div>
                    <div class="arch-title">📊 Dixon-Coles Poisson</div>
                    <div class="arch-subtitle">從球隊攻防強度推導期望進球 λ</div>
                </div>
                <span class="arch-weight-chip">外層權重 40%</span>
            </div>
            <div class="arch-formula">λ_A = atk_A × vul_B / μ</div>
            <div class="arch-content">
                <div>
                    <div class="arch-block-title">📐 變數定義</div>
                    <div class="arch-block-body">
                        • <b>μ</b> = 1.35（國際賽進球均值）<br>
                        • <b>atk</b> = 場均進球 × 足聯係數<br>
                        • <b>vul</b> = 場均失球 ÷ √足聯係數<br>
                        • 從 λ 積分 Poisson 矩陣得 P(W/D/L)
                    </div>
                </div>
                <div>
                    <div class="arch-block-title">🌐 足聯係數 (cs)</div>
                    <div class="arch-block-body">
                        CONMEBOL 0.97 · UEFA 0.96<br>
                        CONCACAF 0.88 · AFC 0.84<br>
                        CAF 0.78 · OFC 0.72<br>
                        <i style="color:#6b6760">弱聯盟攻擊打折、防守實際更弱</i>
                    </div>
                </div>
            </div>
        </div>
        <div class="arch-arrow">↓</div>
        """,
        unsafe_allow_html=True,
    )

    # ── Layer 3: λ 多因素修正 ──
    st.markdown(
        """
        <div class="arch-layer">
            <div class="arch-layer-header">
                <span class="arch-badge">L3</span>
                <div>
                    <div class="arch-title">⭐ λ 多因素複合修正</div>
                    <div class="arch-subtitle">在純 DC λ 基礎上疊加 4 個現實因子</div>
                </div>
                <span class="arch-weight-chip">乘法修正</span>
            </div>
            <div class="arch-formula">λ_final = λ_DC × Π (rank · squad · form · exp)</div>
            <table class="arch-table">
                <tr><th style="width:30%">因素</th><th style="width:25%">次方</th><th>意義</th></tr>
                <tr><td>🥇 主將 OVR</td><td class="pow">^0.35</td><td>陣容明星密度（權重最大）</td></tr>
                <tr><td>🥈 FIFA 積分</td><td class="pow">^0.22</td><td>歷史排名差距</td></tr>
                <tr><td>🥉 近 2 年勝率</td><td class="pow">^0.18</td><td>球隊近期狀態</td></tr>
                <tr><td>🏅 淘汰賽經驗</td><td class="pow">^0.08</td><td>大賽 DNA（影響最小）</td></tr>
            </table>
            <div class="arch-tip">
                💡 <b>範例</b>：主將 OVR 85 vs 79 → (85/79)<sup>0.35</sup> ≈ <b style="color:#c96442">1.027</b>（λ 提升 +2.7%）
            </div>
        </div>
        <div class="arch-arrow">↓</div>
        """,
        unsafe_allow_html=True,
    )

    # ── Layer 4: 融合 + Monte Carlo ──
    st.markdown(
        """
        <div class="arch-layer">
            <div class="arch-layer-header">
                <span class="arch-badge">L4</span>
                <div>
                    <div class="arch-title">🎲 融合機率 + Monte Carlo 模擬</div>
                    <div class="arch-subtitle">合併 L1 與 L2 機率 + 賽程級模擬</div>
                </div>
                <span class="arch-weight-chip">最終輸出</span>
            </div>
            <div class="arch-content">
                <div>
                    <div class="arch-block-title">🎯 MAP 比分（方向約束）</div>
                    <div class="arch-block-body">
                        1. 依 P(W/D/L) 取主預測方向<br>
                        2. 在該方向 cells 找最強 Poisson cell<br>
                        3. ★ 標記該比分<br>
                        <i style="color:#5f8466">→ 避免「20% 平局卻每場都平」悖論</i>
                    </div>
                </div>
                <div>
                    <div class="arch-block-title">🎲 Monte Carlo 奪冠模擬</div>
                    <div class="arch-block-body">
                        • 10,000 次完整賽程模擬<br>
                        • 小組賽 → 32 強淘汰賽<br>
                        • 主將陣容 OVR 修正（與單場一致）<br>
                        • 累計奪冠次數 / 10,000 = 機率<br>
                        <i style="color:#6b6760">→ 約 1,040,000 場虛擬比賽</i>
                    </div>
                </div>
            </div>
        </div>
        <div class="arch-arrow">↓</div>
        <div class="arch-output">
            <div class="arch-output-label">FINAL OUTPUT · 最終預測</div>
            <div class="arch-output-content">
                <code>P(勝/平/負) = 0.60 × XGB + 0.40 × Poisson(λ)</code><br>
                <span style="font-size:13px;color:#fff;opacity:0.9">＋ 預測比分　＋　奪冠機率</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 權重結構總覽 ──
    st.markdown("")
    st.markdown("##### 📐 完整權重結構速查")
    weights_summary = pd.DataFrame([
        ['外層融合', 'XGBoost 分類器', '60%', 'P(W/D/L) 加權平均的權重'],
        ['外層融合', 'Dixon-Coles Poisson', '40%', 'P(W/D/L) 加權平均的權重'],
        ['λ 修正', '主將 OVR', '^0.35', '5 名主將綜合能力比次方'],
        ['λ 修正', 'FIFA 積分', '^0.22', '兩隊 FIFA 積分比次方'],
        ['λ 修正', '近 2 年勝率', '^0.18', '近期狀態勝率比次方'],
        ['λ 修正', '淘汰賽經驗', '^0.08', '歷屆世界盃淘汰賽場次比次方'],
        ['足聯校正', 'atk 縮放', '× cs', '弱聯盟攻擊打折（CONMEBOL=0.97, OFC=0.72）'],
        ['足聯校正', 'vul 縮放', '÷ √cs', '弱聯盟防守上調（實際對強隊更易失球）'],
    ], columns=['類別', '因素', '權重', '說明'])
    st.dataframe(weights_summary, use_container_width=True, hide_index=True)
    st.caption(
        "📖 **解讀**：外層融合決定「機率分布」；λ 修正決定「期望進球」；"
        "足聯校正消除「跨聯盟比較偏差」。三層獨立但層層遞進，避免單一信號主導。"
    )

# ============================================================
# PAGE 2: 2026 預測
# ============================================================
elif page == "🔮 2026 預測":
    st.title("🔮 2026 世界盃比分預測")
    st.markdown("**XGBoost(60%) + Dixon-Coles(40%) · λ 加權：主將 OVR · FIFA 積分 · 近期狀態 · 淘汰賽經驗 · Walk-Forward 驗證**")
    st.markdown("---")

    match_df = load_match_data()
    fifa_df  = load_fifa_ranking()

    # 先試讀預訓練 pkl（< 1 秒）；若不存在則 fallback 到即時訓練
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
                score1_color, score2_color = '#c96442', '#9b958a'
            elif r['goal1'] < r['goal2']:
                score1_color, score2_color = '#9b958a', '#c96442'
            else:
                score1_color = score2_color = '#b58a3b'

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
                    # Claude 主題配色：珊瑚橘 vs 靛藍
                    C1 = _CLAUDE_ACCENT   # 球隊1（Claude 珊瑚橘）
                    C2 = _CLAUDE_ACCENT2  # 球隊2（暖靛藍）

                    fig_bar = go.Figure()
                    cats = [m[0] for m in metrics_list]
                    v1s = [m[1] for m in metrics_list]
                    v2s = [m[2] for m in metrics_list]
                    fig_bar.add_trace(go.Bar(
                        name=info1['cn'], x=cats, y=v1s, marker=dict(color=C1),
                    ))
                    fig_bar.add_trace(go.Bar(
                        name=info2['cn'], x=cats, y=v2s, marker=dict(color=C2),
                    ))
                    fig_bar.update_layout(**claude_layout(
                        barmode='group', height=280,
                        margin=dict(l=10, r=10, t=40, b=40),
                        legend=dict(orientation='h', y=1.15,
                                    font=dict(color=_CLAUDE_TEXT, size=13)),
                    ))
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
                        colorscale=[[0, '#faf9f5'], [0.5, '#f3e9e4'], [1, '#c96442']],
                        showscale=False,
                        text=text_z, texttemplate='%{text}', textfont=dict(size=10),
                    ))
                    fig_h.update_layout(**claude_layout(
                        xaxis_title=f'{info2["cn"]} 進球',
                        yaxis_title=f'{info1["cn"]} 進球',
                        height=280, margin=dict(l=50, r=10, t=10, b=40),
                    ))
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
    st.markdown(
        "**49,257 場真實國際比賽（1872-2026）** ｜ 本頁透過歷史資料驗證"
        "我們的模型假設：為什麼選 Poisson、為什麼用近 8 年、為什麼平局難預測。"
    )
    st.markdown("---")

    match_df = load_match_data()
    wc = match_df[match_df['tournament'].str.contains('World Cup', na=False)]

    # ============================================================
    # 區塊 1：進球分佈 + Poisson 對比
    # ============================================================
    st.subheader("📊 區塊一：世界盃進球分佈 — Poisson 假設驗證")
    st.markdown(
        "**為什麼選擇 Poisson 模型？** 足球進球是「稀疏隨機事件」典型範例："
        "射門次數高但成功率低，符合 Poisson 分佈的數學前提。"
    )

    c1, c2 = st.columns(2)
    with c1:
        all_g = pd.concat([wc['home_score'], wc['away_score']])
        # 計算實際分佈和 Poisson 理論分佈
        from scipy.stats import poisson as scipy_poisson
        mean_goals = float(all_g.mean())
        max_g = min(7, int(all_g.max()))
        actual_dist = all_g[all_g <= max_g].value_counts(normalize=True).sort_index()
        theoretical_dist = [scipy_poisson.pmf(k, mean_goals) for k in range(max_g + 1)]
        compare_df = pd.DataFrame({
            '進球數': list(range(max_g + 1)),
            '實際比例': [actual_dist.get(k, 0) for k in range(max_g + 1)],
            'Poisson理論': theoretical_dist,
        })
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(name='實際比例', x=compare_df['進球數'],
                                   y=compare_df['實際比例'], marker_color='#3366cc'))
        fig_dist.add_trace(go.Scatter(name=f'Poisson(λ={mean_goals:.2f})',
                                       x=compare_df['進球數'], y=compare_df['Poisson理論'],
                                       mode='lines+markers', line=dict(color='#f7c948', width=3),
                                       marker=dict(size=10)))
        fig_dist.update_layout(
            title=f"世界盃單隊進球分佈（{len(all_g):,} 個樣本）",
            xaxis_title='單場進球數', yaxis_title='出現比例',
            height=380, legend=dict(x=0.55, y=0.95),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with c2:
        st.markdown("##### 💡 重點解讀")
        # 關鍵統計
        zero_pct = (all_g == 0).mean()
        one_pct = (all_g == 1).mean()
        two_pct = (all_g == 2).mean()
        rare_pct = (all_g >= 4).mean()
        st.markdown(
            f"**關鍵統計（{len(all_g):,} 個單隊進球記錄）：**\n\n"
            f"- 進 0 球：**{zero_pct:.1%}** — 進攻無功而返的常態\n"
            f"- 進 1 球：**{one_pct:.1%}** — 最高頻次\n"
            f"- 進 2 球：**{two_pct:.1%}** — 第二高頻次\n"
            f"- 進 ≥4 球：**{rare_pct:.1%}** — 罕見爆發\n\n"
            f"**平均進球（λ）：{mean_goals:.2f}**\n\n"
            "---\n\n"
            "**📌 為什麼這對模型重要？**\n"
            "1. 藍色柱（實際）幾乎貼合金線（Poisson 理論）→ "
            "證實足球進球符合 Poisson 分佈\n"
            "2. **這就是我們用 `Poisson(λ)` 計算每個比分機率的數學基礎**\n"
            "3. 進球數高度集中在 0-2 球 → 1-1、1-0、2-1 是最常見比分\n"
            "4. 4+ 球極罕見 → 模型 0-7 進球範圍涵蓋 99.9%+ 機率"
        )

    st.markdown("---")

    # ============================================================
    # 區塊 2：進球趨勢
    # ============================================================
    st.subheader("📈 區塊二：每場平均進球趨勢 — 為何只用近 8 年？")
    st.markdown(
        "**現代足球與歷史足球差距巨大**：戰術、體能、規則持續演化。"
        "用 1950s 的進球數據預測 2026 會嚴重失準。"
    )

    yearly = match_df[match_df['year'] >= 1970].groupby('year').agg(
        total=('home_score', lambda x: x.sum() + match_df.loc[x.index, 'away_score'].sum()),
        count=('home_score', 'count')
    )
    yearly['avg'] = yearly['total'] / yearly['count']
    yearly = yearly.reset_index()
    # 加 8 年滾動平均
    yearly['avg_8yr'] = yearly['avg'].rolling(window=8, center=True).mean()

    c3, c4 = st.columns([2, 1])
    with c3:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(name='年均進球', x=yearly['year'], y=yearly['avg'],
                                        mode='lines+markers', line=dict(color='#3366cc', width=1.5),
                                        marker=dict(size=5), opacity=0.6))
        fig_trend.add_trace(go.Scatter(name='8 年滾動平均', x=yearly['year'], y=yearly['avg_8yr'],
                                        mode='lines', line=dict(color='#e94560', width=3)))
        fig_trend.update_layout(
            title="每場平均總進球（1970-2025）",
            xaxis_title='年份', yaxis_title='場均總進球',
            height=380, legend=dict(x=0.02, y=0.05),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with c4:
        st.markdown("##### 💡 重點解讀")
        recent_avg = yearly[yearly['year'] >= 2018]['avg'].mean()
        old_avg = yearly[(yearly['year'] >= 1970) & (yearly['year'] < 1990)]['avg'].mean()
        st.markdown(
            f"**1970-1989 平均**：{old_avg:.2f} 球/場\n\n"
            f"**2018-2025 平均**：{recent_avg:.2f} 球/場\n\n"
            f"**差距**：{(recent_avg - old_avg):+.2f} 球\n\n"
            "---\n\n"
            "**📌 模型設計決策**\n"
            f"- `compute_team_strength()` 只取最近 **8 年** 資料\n"
            f"- 加入 **時間衰減權重** `exp(-0.1 × 年數)`：\n"
            f"  - 1 年前的權重 = 0.90\n"
            f"  - 5 年前 = 0.61\n"
            f"  - 8 年前 = 0.45\n"
            f"- → 越近期的比賽對 λ 影響越大\n"
            f"- → 避免用過時數據誤判球隊現況"
        )

    st.markdown("---")

    # ============================================================
    # 區塊 3：球隊實力分組
    # ============================================================
    st.subheader("🏆 區塊三：各組球隊實力對比 — 死亡之組在哪？")
    st.markdown(
        "**各組強度不均** 是世界盃常態。下表按組別排序，"
        "可一眼看出哪些組擁有多支強隊（死亡之組），哪些組相對輕鬆。"
    )

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
                '場次': s['matches'],
                '_win': s['win_rate'],
            })

    hist_df_full = pd.DataFrame(team_hist)
    # 三種分類指標：
    #   - 平均勝率：整體強度（死亡之組 = 所有隊都強）
    #   - 強弱差距 (max-min)：勝率分布（強弱懸殊組 = 一隊獨大）
    #   - 強隊數 (>50%)：高勝率隊伍計數
    group_stats = hist_df_full.groupby('組別').agg(
        平均勝率=('_win', 'mean'),
        最強隊勝率=('_win', 'max'),
        最弱隊勝率=('_win', 'min'),
        強隊數=('_win', lambda x: (x > 0.50).sum()),
    ).reset_index()
    group_stats['強弱差距'] = group_stats['最強隊勝率'] - group_stats['最弱隊勝率']

    c5, c6 = st.columns([1, 1])
    with c5:
        st.markdown("**📋 各隊歷史實力（按組分類）**")
        hist_df = hist_df_full.drop('_win', axis=1).sort_values(['組別', '勝率'], ascending=[True, False])
        st.dataframe(hist_df, use_container_width=True, hide_index=True, height=400)

    with c6:
        st.markdown("**🔥 各組強度與懸殊度**")
        group_stats_display = group_stats.copy()
        for col in ['平均勝率', '最強隊勝率', '最弱隊勝率', '強弱差距']:
            group_stats_display[col] = group_stats_display[col].apply(lambda x: f"{x:.1%}")
        # 按平均勝率排序顯示
        group_stats_display = group_stats_display.sort_values('平均勝率', ascending=False)
        st.dataframe(
            group_stats_display[['組別', '平均勝率', '強弱差距', '強隊數', '最強隊勝率']],
            use_container_width=True, hide_index=True, height=400
        )

    # 死亡之組：平均勝率最高（所有隊都強）
    deadly_groups = group_stats.sort_values('平均勝率', ascending=False).head(3)['組別'].tolist()
    # 強弱懸殊組：強弱差距最大（一隊獨大 → 強隊好出線）
    lopsided_groups = group_stats.sort_values('強弱差距', ascending=False).head(3)['組別'].tolist()
    st.markdown(
        f"##### 💡 解讀（兩種「易出線」概念）\n"
        f"- **🔥 死亡之組（強隊密集）**：Group **{', '.join(deadly_groups)}**\n"
        f"  - 判定依據：**平均勝率最高**（所有隊都是強隊）\n"
        f"  - 特性：出線競爭最激烈，黑馬機率高\n\n"
        f"- **⚖️ 強弱懸殊組（一隊獨大）**：Group **{', '.join(lopsided_groups)}**\n"
        f"  - 判定依據：**強弱差距最大**（max勝率 − min勝率）\n"
        f"  - 特性：強隊面對弱隊容易壓制，前 2 名出線早早鎖定\n\n"
        f"- **📌 為何分兩種定義？**\n"
        f"  - 「平均勝率低」≠ 強隊容易出線（可能整組都很爛沒人脫穎而出）\n"
        f"  - 真正「好走」的組是「一強三弱」型，強弱差距大才是關鍵\n\n"
        f"- **🎲 模型意義**：Monte Carlo 模擬 10,000 次小組賽時，"
        f"死亡之組的「黑馬出線」次數明顯偏高；強弱懸殊組的前 2 名出線機率則高度集中。"
    )

    # ── 模型評估區塊（v2.3 新增）──
    st.markdown("---")
    st.subheader("🔬 區塊四：XGBoost 模型評估（2022 WC 測試集）")
    st.markdown(
        "**這裡回答一個關鍵問題：你的模型到底準不準？** "
        "我們用 2022 卡達世界盃小組賽當「未見過」的測試集，"
        "5 個面向徹底評估模型表現：整體準確率、各類別表現、機率可信度、特徵貢獻、統計顯著性。"
    )
    em = load_eval_metrics()
    if em is None:
        st.info("⚠️ 找不到 models/eval_metrics.pkl，請先在本機跑 `python pretrain.py` 產生評估數據。")
    else:
        _acc = em['accuracy']
        _n = em['n_test']
        # 概覽 metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("測試集場數", f"{_n} 場", help="2022 卡達世界盃小組賽全部比賽")
        with col_b:
            st.metric("整體準確率", f"{_acc:.1%}",
                      delta=f"+{_acc-0.333:.1%} vs 隨機 33.3%",
                      help="三分類問題中，模型答對的比例")
        with col_c:
            st.metric("隨機猜測基準", "33.3%",
                      help="三向（勝/平/負）隨機猜測的理論準確率")
        st.markdown("")
        tab_cm, tab_roc, tab_cal, tab_fi, tab_fisher = st.tabs(
            ["📊 混淆矩陣", "📈 ROC 曲線", "🎚 校準曲線", "🔍 特徵重要性", "🧪 費雪檢定"]
        )

        # ── Tab 1: Confusion Matrix ──
        with tab_cm:
            cm_data = em['cm']
            cm_arr_local = np.array(cm_data)
            # 中文化標籤
            label_map_x = ['預測：主隊負', '預測：平局', '預測：主隊勝']
            label_map_y = ['實際：主隊負', '實際：平局', '實際：主隊勝']
            import plotly.graph_objects as _go
            # row-wise 百分比
            cm_norm = [[round(v / max(sum(row), 1) * 100, 1) for v in row] for row in cm_data]
            # 每格加上 ✓/✗ + 場數 + 百分比
            text_vals = []
            for i in range(3):
                row_txt = []
                for j in range(3):
                    mark = '✓ 正確' if i == j else '✗ 誤判'
                    row_txt.append(
                        f"<b>{mark}</b><br>{cm_data[i][j]} 場 ({cm_norm[i][j]}%)"
                    )
                text_vals.append(row_txt)
            # 自訂色階：對角線深珊瑚橘，誤判格淺灰
            fig_cm = _go.Figure(data=_go.Heatmap(
                z=[[cm_data[i][j] for j in range(3)] for i in range(3)],
                x=label_map_x, y=label_map_y,
                colorscale=[[0, '#faf9f5'], [0.3, '#f3e9e4'],
                            [0.7, '#e0a48a'], [1, '#c96442']],
                text=text_vals,
                texttemplate="%{text}",
                textfont={"size": 13, "color": "#1f1e1c"},
                showscale=False,
                hovertemplate='%{y}<br>%{x}<br>場數: %{z}<extra></extra>',
            ))
            fig_cm.update_layout(**claude_layout(
                xaxis_title="↓ 模型預測（X 軸）",
                yaxis_title="實際結果（Y 軸） →",
                height=420,
                margin=dict(l=110, r=20, t=20, b=70),
            ))
            # Y 軸反轉，讓主隊勝在上方更直觀
            fig_cm.update_yaxes(autorange='reversed')
            st.plotly_chart(fig_cm, use_container_width=True)

            # 各類別 Recall + 直觀解釋
            lose_recall = cm_arr_local[0, 0] / max(cm_arr_local[0, :].sum(), 1)
            draw_recall = cm_arr_local[1, 1] / max(cm_arr_local[1, :].sum(), 1)
            win_recall = cm_arr_local[2, 2] / max(cm_arr_local[2, :].sum(), 1)

            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("主隊勝 Recall", f"{win_recall:.0%}",
                          help=f"實際主隊勝 {int(cm_arr_local[2,:].sum())} 場，"
                               f"模型答對 {int(cm_arr_local[2,2])} 場")
            with col_r2:
                st.metric("平局 Recall", f"{draw_recall:.0%}",
                          delta="難預測",
                          delta_color="off",
                          help=f"實際平局 {int(cm_arr_local[1,:].sum())} 場，"
                               f"模型答對 {int(cm_arr_local[1,1])} 場")
            with col_r3:
                st.metric("主隊負 Recall", f"{lose_recall:.0%}",
                          help=f"實際主隊負 {int(cm_arr_local[0,:].sum())} 場，"
                               f"模型答對 {int(cm_arr_local[0,0])} 場")

            with st.expander("📖 詳細解讀（給第一次看混淆矩陣的人）", expanded=False):
                st.markdown(
                    f"""
**🎯 這張表在做什麼？**

把 2022 卡達世界盃小組賽 **{int(cm_arr_local.sum())} 場**全部丟給訓練好的模型猜，
然後用一張表記錄「模型猜得準不準」。

**📊 怎麼讀？**
- **每一橫列**＝實際比賽結果（共 3 種：主隊勝 / 平局 / 主隊負）
- **每一直行**＝模型猜的結果
- 兩者交叉的格子，就是「實際 A、模型猜 B」的場次數

**✅ 預測正確（對角線）**
- 主隊勝 → 模型猜勝：**{int(cm_arr_local[2,2])} 場**
- 平局 → 模型猜平：**{int(cm_arr_local[1,1])} 場**
- 主隊負 → 模型猜負：**{int(cm_arr_local[0,0])} 場**
- 三者合計：**{int(cm_arr_local[0,0]+cm_arr_local[1,1]+cm_arr_local[2,2])} 場**（{(cm_arr_local[0,0]+cm_arr_local[1,1]+cm_arr_local[2,2])/cm_arr_local.sum():.1%}）

**❌ 預測錯誤（非對角線）**
模型誤判最常發生在「實際平局卻被預測為主隊勝/負」與「實際主隊負卻被預測為主隊勝」。

**📐 Recall（召回率）怎麼算？**

`Recall = 該類別答對 / 該類別實際發生總場數`

例如「主隊勝 Recall = {win_recall:.0%}」的意思是：
**真的是主隊贏球的場次中，模型成功抓出了 {win_recall:.0%}**。

**💡 為什麼平局 Recall 最低（{draw_recall:.0%}）？**

平局發生在兩隊實力**接近**時，模型看到「勢均力敵」的特徵反而會傾向「強勢方獲勝」。
這是足球預測的**世界性難題** — 連 Bet365 / Pinnacle 等專業博弈公司也都有同樣問題。

因此本系統的「平局機率」是**參考值**，不會用平局當主預測。
"""
                )

        # ── Tab 2: ROC 曲線 ──
        with tab_roc:
            import plotly.graph_objects as _go2
            class_cn_roc = {'Team1 Lose': '主隊負', 'Draw': '平局', 'Team1 Win': '主隊勝'}
            colors_roc = [_CLAUDE_ACCENT, '#b58a3b', _CLAUDE_ACCENT2]
            fig_roc = _go2.Figure()
            auc_summary = []
            for idx, (name, rd) in enumerate(em['roc'].items()):
                cn = class_cn_roc.get(name, name)
                fig_roc.add_trace(_go2.Scatter(
                    x=rd['fpr'], y=rd['tpr'],
                    mode='lines',
                    name=f"{cn} (AUC = {rd['auc']:.2f})",
                    line=dict(width=2.5, color=colors_roc[idx % 3]),
                ))
                auc_summary.append((name, rd['auc']))
            fig_roc.add_trace(_go2.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                line=dict(dash='dash', color='#9b958a', width=1),
                name='隨機基準 (AUC=0.5)', showlegend=True,
            ))
            fig_roc.update_layout(**claude_layout(
                xaxis_title="假陽性率 (FPR)",
                yaxis_title="真陽性率 (TPR)",
                height=420,
                legend=dict(x=0.50, y=0.08,
                            bgcolor='rgba(255,255,255,0.85)',
                            bordercolor=_CLAUDE_GRID, borderwidth=1),
            ))
            st.plotly_chart(fig_roc, use_container_width=True)

            # AUC 評等 — 三欄 metric 卡
            def _rating(v):
                if v >= 0.8: return "🟢 優秀"
                if v >= 0.7: return "🟡 良好"
                if v >= 0.6: return "🟠 可接受"
                return "🔴 偏弱"
            cols_auc = st.columns(len(auc_summary))
            for col, (name, auc_v) in zip(cols_auc, auc_summary):
                cn = class_cn_roc.get(name, name)
                with col:
                    st.metric(f"{cn} AUC", f"{auc_v:.2f}", delta=_rating(auc_v),
                              delta_color="off")

            with st.expander("📖 詳細解讀（ROC 曲線是什麼？）", expanded=False):
                auc_pairs = [(class_cn_roc.get(n, n), v) for n, v in auc_summary]
                auc_lines = "\n".join(
                    f"- **{cn}** AUC = {v:.2f} {_rating(v)}" for cn, v in auc_pairs
                )
                st.markdown(
                    f"""
**🎯 ROC 曲線在量什麼？**

ROC 全名 **Receiver Operating Characteristic**（接收者操作特徵曲線）。
它衡量的不是「猜對幾場」，而是「**模型把機率排序得對不對**」。

**比喻**：如果我把 100 場比賽，依照模型給的「主隊勝機率」由高到低排序，
真正主隊勝的場次有沒有集中在前段？如果有 → 模型排序能力強 → 曲線靠左上方。

**📐 兩個座標軸的意思**

- **X 軸（假陽性率 FPR）**：本來不會發生的，模型卻誤判會發生的比例
- **Y 軸（真陽性率 TPR）**：本來會發生的，模型成功抓出來的比例
- 兩軸都 0~1，理想模型走左上角（FPR=0, TPR=1）

**📊 AUC（曲線下面積）的意思**

| AUC | 鑑別力 | 解讀 |
|-----|--------|------|
| 0.90+ | 出色 | 運動預測極少達到 |
| 0.80–0.89 | 優秀 | 商用模型水準 |
| 0.70–0.79 | 良好 | 有明顯預測能力 |
| 0.60–0.69 | 可接受 | 比隨機好，但不穩 |
| < 0.60 | 偏弱 | 接近瞎猜 |

**📋 本模型各類別 AUC：**

{auc_lines}

**💡 為什麼平局 AUC 通常最低？**

平局沒有清晰的「特徵指紋」— 強強對決可能踢平、弱弱對決也可能踢平。
模型很難用單一特徵集穩定鎖定平局。所有運動預測模型都面臨此限制。
"""
                )

        # ── Tab 3: Calibration ──
        with tab_cal:
            import plotly.graph_objects as _go3
            cal = em['calibration']
            fig_cal = _go3.Figure()
            fig_cal.add_trace(_go3.Scatter(
                x=cal['prob_pred'], y=cal['prob_true'],
                mode='lines+markers',
                name='XGBoost（主隊勝）',
                line=dict(color=_CLAUDE_ACCENT, width=2.5),
                marker=dict(size=10),
            ))
            fig_cal.add_trace(_go3.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                line=dict(dash='dash', color='#9b958a', width=1),
                name='完美校準',
            ))
            fig_cal.update_layout(**claude_layout(
                xaxis_title="預測機率（模型說有 X% 機率主隊贏）",
                yaxis_title="實際頻率（這些場次中真的有 Y% 主隊贏）",
                height=420,
                xaxis=dict(range=[0, 1], gridcolor=_CLAUDE_GRID,
                           linecolor=_CLAUDE_GRID, color=_CLAUDE_TEXT, zeroline=False),
                yaxis=dict(range=[0, 1], gridcolor=_CLAUDE_GRID,
                           linecolor=_CLAUDE_GRID, color=_CLAUDE_TEXT, zeroline=False),
                legend=dict(x=0.55, y=0.10,
                            bgcolor='rgba(255,255,255,0.85)',
                            bordercolor=_CLAUDE_GRID, borderwidth=1),
            ))
            st.plotly_chart(fig_cal, use_container_width=True)
            # 計算 ECE 並用 metric 卡顯示
            try:
                import numpy as _np
                pred_arr = _np.array(cal['prob_pred'])
                true_arr = _np.array(cal['prob_true'])
                ece = float(_np.mean(_np.abs(pred_arr - true_arr)))
                if ece < 0.05:
                    rating_label = "🟢 校準極佳"
                elif ece < 0.10:
                    rating_label = "🟡 校準良好"
                else:
                    rating_label = "🟠 偏差較大"
            except Exception:
                ece = None; rating_label = "—"
            cc1, cc2 = st.columns(2)
            with cc1:
                st.metric("預期校準誤差 (ECE)", f"{ece:.3f}" if ece is not None else "—",
                          delta=rating_label, delta_color="off")
            with cc2:
                st.metric("目標值", "0.000",
                          help="ECE 越接近 0 表示預測機率越接近真實發生機率")

            with st.expander("📖 詳細解讀（校準曲線是什麼？）", expanded=False):
                st.markdown(
                    """
**🎯 校準曲線在驗證什麼？**

校準曲線（Calibration Plot）檢驗「**機率本身可不可信**」。
當模型說「主隊有 70% 勝率」時，這 70% 是真實機率，還是隨便喊的數字？

**📊 怎麼看圖**

| 紅線位置 | 意義 | 範例 |
|----------|------|------|
| **貼合虛線** | 完美校準 — 字面值可信 | 喊 70%，真的 70% 場次主隊贏 |
| **在虛線上方** | 保守 — 低估勝率 | 喊 60%，實際 75% 場次主隊贏 |
| **在虛線下方** | 過度自信 — 高估勝率 | 喊 80%，實際只有 65% 場次主隊贏 |

**📐 ECE（預期校準誤差）怎麼算？**

`ECE = 平均(|預測機率 − 實際發生頻率|)`

- ECE = 0.000 表示完美校準
- ECE < 0.05 表示優秀（商用等級）
- ECE > 0.10 表示機率值不能按字面信賴

**💡 為什麼校準很重要？**

對於 Monte Carlo 模擬 10,000 次比賽來說，**機率的「絕對水平」必須準**。

舉例：如果模型把所有 70% 勝率都喊成 90%，10,000 次模擬出來的奪冠分布會嚴重偏向強隊，
弱隊根本沒機會出線。校準曲線就是用來驗證「機率值能不能直接拿來算」。
"""
                )

        # ── Tab 4: Feature Importance ──
        with tab_fi:
            import plotly.graph_objects as _go4
            fi = em['feature_importance']
            # 自動判斷數值範圍決定精度（gain 通常 0.0-1.0，需要 3 位小數）
            _max_val = max(fi['values']) if fi['values'] else 1.0
            _fmt = "{:.3f}" if _max_val < 1.0 else "{:.2f}" if _max_val < 10 else "{:.1f}"
            fig_fi = _go4.Figure(_go4.Bar(
                x=fi['values'], y=fi['features'],
                orientation='h',
                marker_color=_CLAUDE_ACCENT,
                text=[_fmt.format(v) for v in fi['values']],
                textposition='outside',
                textfont=dict(color=_CLAUDE_TEXT),
            ))
            fig_fi.update_layout(**claude_layout(
                xaxis_title="重要性分數（gain）",
                height=max(350, len(fi['features']) * 32),
                margin=dict(l=180, r=40, t=20, b=40),
            ))
            st.plotly_chart(fig_fi, use_container_width=True)

            # Top 3 特徵
            top_features = sorted(zip(fi['features'], fi['values']),
                                  key=lambda x: x[1], reverse=True)[:3]
            feat_translation = {
                'rank_diff': 'FIFA 積分差', 'pts_diff': 'FIFA 積分差',
                'win_rate_diff': '勝率差', 'avg_goals_diff': '場均進球差',
                'avg_conceded_diff': '場均失球差', 'goal_diff_diff': '淨勝球差',
                'matches_diff': '出賽場數差', 'rank1': '球隊1 FIFA 積分',
                'rank2': '球隊2 FIFA 積分', 'fifa_pts_diff': 'FIFA 積分差',
                'confed_diff': '足聯差', 'recent_form_diff': '近期狀態差',
            }
            cols_top = st.columns(3)
            medals = ['🥇', '🥈', '🥉']
            _total_gain = sum(fi['values']) if fi['values'] else 1.0
            for (name, val), col, medal in zip(top_features, cols_top, medals):
                cn = feat_translation.get(name, name)
                share = val / _total_gain if _total_gain > 0 else 0
                with col:
                    st.metric(f"{medal} {cn}", _fmt.format(val),
                              delta=f"佔總 gain {share:.1%}",
                              delta_color="off",
                              help=f"原始特徵名：{name}")

            with st.expander("📖 詳細解讀（特徵重要性怎麼讀？）", expanded=False):
                top_lines = "\n".join(
                    f"- {m} **{feat_translation.get(n, n)}** — gain {_fmt.format(v)}"
                    f" (佔 {v/_total_gain:.1%})"
                    for (n, v), m in zip(top_features, medals)
                )
                st.markdown(
                    f"""
**🎯 圖表在說什麼？**

XGBoost 是由很多決策樹組成的模型，每棵樹用「特徵」來分裂節點做判斷。
**特徵重要性（Feature Importance）** 衡量每個特徵在所有樹的分裂節點中
「提供了多少資訊增益（gain）」的總和。

**白話：分數越高 = 模型越靠這個特徵做判斷。**

**🏆 本模型 Top 3 最有影響力的特徵**

{top_lines}

**💡 為什麼「FIFA 積分差」通常排第一？**

FIFA 排名濃縮了一支球隊「過去 4 年所有國際賽」的結果，
是公開資料中**最濃縮的綜合實力指標**。
兩隊積分差距越大，比賽結果越容易預測 → 模型最常用這個特徵分裂。

**🤔 低分特徵代表沒用嗎？**

**不一定**。XGBoost 的 gain 計算會把高度相關的特徵分散權重 —

例如「勝率差」和「FIFA 積分差」本質上量測相似的東西（強隊勝率高），
模型只會主要依賴其中一個（通常是 rank_diff），
另一個的 gain 自然就低了，但**並不代表該特徵沒資訊**。

**📌 怎麼運用這個資訊？**

1. **建模啟發**：低重要性 + 高相關 → 可考慮移除以簡化模型
2. **業務理解**：知道模型依靠的核心信號是什麼，方便向他人解釋
3. **新特徵嘗試**：若 Top 3 都是排名類特徵，可考慮加入「球員陣容、傷兵」等
   排名以外的資訊提升模型多樣性
"""
                )

        # ── Tab 5: Fisher's Exact Test ──
        with tab_fisher:
            from scipy.stats import fisher_exact, chi2_contingency
            cm_arr = np.array(em['cm'])
            total_n = int(cm_arr.sum())
            class_cn = {'Team1 Lose': '主隊負', 'Draw': '平局', 'Team1 Win': '主隊勝'}

            # 主表：三類別費雪檢定結果
            fisher_rows = []
            fisher_raw = []
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
                fisher_raw.append({'label': class_cn.get(label, label),
                                   'p': p_value, 'or': odds_ratio,
                                   'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})
            fisher_df = pd.DataFrame(fisher_rows)
            st.dataframe(fisher_df, use_container_width=True, hide_index=True)

            # 整體卡方檢定 3 個 metric 卡
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
                st.metric("整體 p-value", f"{chi_p:.4f}" if not np.isnan(chi_p) else '—',
                          delta="✅ 顯著" if chi2_ok else "⚠️ 未達顯著",
                          delta_color="off")

            with st.expander("📖 詳細解讀（費雪檢定是什麼？）", expanded=False):
                # 動態組成各類別解讀
                sig_lines = []
                for r in fisher_raw:
                    if np.isnan(r['p']):
                        verdict = '— 無法計算'
                    elif r['p'] < 0.001:
                        verdict = f"**p = {r['p']:.4f} → 極顯著 ✓✓✓**（極強證據模型有效）"
                    elif r['p'] < 0.01:
                        verdict = f"**p = {r['p']:.4f} → 高度顯著 ✓✓**（強證據模型有效）"
                    elif r['p'] < 0.05:
                        verdict = f"**p = {r['p']:.4f} → 顯著 ✓**（達到統計顯著門檻）"
                    else:
                        verdict = f"p = {r['p']:.4f} → 未達顯著 ✗（不能拒絕「隨機猜測」假設）"
                    or_str = f"，OR = {r['or']:.2f}" if not np.isnan(r['or']) else ''
                    sig_lines.append(f"- **{r['label']}** {verdict}{or_str}")

                st.markdown(
                    f"""
**🎯 為什麼要做這個檢定？**

光看「正確率 55%」這個數字，無法回答：
**「這 55% 是真的有預測能力，還是純粹運氣好剛好猜對？」**

費雪精確檢定（Fisher's Exact Test）就是用嚴格的統計方法，
**證明模型的預測表現「不是運氣使然」**。

**🧪 怎麼運作？**

由於模型輸出有 3 類（勝/平/負），費雪原版只能處理 2×2 表，
我們用 **「一對多（One-vs-Rest）」** 策略：

**舉例：檢定「主隊勝」類別**
把所有比賽分成 4 格（2×2 列聯表）：

|  | 模型預測：主隊勝 | 模型預測：其他 |
|---|---|---|
| **實際：主隊勝** | TP（真陽性） | FN（假陰性） |
| **實際：其他** | FP（假陽性） | TN（真陰性） |

對這個 2×2 表做費雪檢定，得到該類別的 p-value 與 Odds Ratio。
三個類別各做一次。

**📐 兩個關鍵指標**

- **p-value**：「假設模型在亂猜，會不會看到這麼好或更好的成績？」
  - p < 0.05 → 不太可能是亂猜（拒絕虛無假設）✓
  - p ≥ 0.05 → 結果可能來自運氣（無法拒絕虛無假設）✗
- **Odds Ratio (OR)**：模型預測該類別「成功的機會比」
  - OR > 1：有正面預測能力（越大越強）
  - OR ≈ 1：跟亂猜差不多
  - OR < 1：比亂猜還差

**🏆 本模型各類別檢定結果**

{chr(10).join(sig_lines)}

**📊 整體卡方檢定（補充）**

χ² = {chi2:.2f}，自由度 = {dof}，p = {chi_p:.4f}
→ {'**整體預測與實際結果有顯著關聯**，證實模型不是亂猜。' if chi2_ok else '**整體未達顯著**，需檢視模型設計或樣本量。'}

**💡 為什麼平局通常不顯著？**

平局發生機率本來就低（世界盃約 25%），加上樣本量小（2022 WC 小組賽只有 {total_n} 場），
就算模型對平局有些預測能力，也很容易因為「統計檢力不足」而 p > 0.05。
這不代表模型對平局完全沒能力，只是**無法從這個樣本量證明它**。
"""
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
        conf = CONFED_MAP.get(selected_team, 'CAF')  # 用 constants.py 單一來源
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

    # ── 近期賽果 + 球員能力卡（小組賽程預測已移除，避免與「🔮 2026 預測」頁重複） ──
    tab_recent, tab_players = st.tabs(["📅 近期賽果", "👤 球員能力卡"])

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
        st.caption(f"📌 {info['cn']} 的 Group {grp} 小組賽預測請至「🔮 2026 預測」頁查看")

    with tab_players:
        players = SQUAD_DATA.get(selected_team, [])
        if not players:
            st.info("此隊球員資料尚未收錄")
        else:
            st.markdown(f"##### {info['cn']} 出賽球員能力卡（共 {len(players)} 人）")
            import streamlit.components.v1 as _components

            # Claude 暖色調 6 屬性配色
            ATTR_COLORS = {
                'pac': '#1f6e8c',  # 速度（暖靛）
                'sho': '#c96442',  # 射門（珊瑚橘）
                'pas': '#5f8466',  # 傳球（橄欖綠）
                'dri': '#b58a3b',  # 盤帶（金棕）
                'def': '#7a5fa7',  # 防守（淡紫）
                'phy': '#a8593e',  # 體能（赤陶）
            }
            ATTR_LABELS = {'pac':'PAC','sho':'SHO','pas':'PAS','dri':'DRI','def':'DEF','phy':'PHY'}

            # SOFIFA 球員 ID 對照表（FIFA 25 / EAFC 25 為主）
            # 重要：每個 ID 必須唯一，否則會顯示錯誤頭像
            # 舊版有重複 ID 已修正：
            #   - H. Kane 與 Son Heung-min 同為 202126 → Kane 暫移除避免錯像
            #   - V. Gyökeres 與 L. Díaz 同為 243726 → Gyökeres 暫移除
            # 未驗證的球員一律 fallback 顯示英文姓名縮寫頭像（不顯示錯誤照片）
            SOFIFA_IDS = {
                # ── 超巨星（高信心 ID）──
                'L. Messi': 158023, 'C. Ronaldo': 20801, 'K. Mbappé': 231747,
                'E. Haaland': 239085, 'K. De Bruyne': 192985, 'Vinícius Jr.': 246169,
                'J. Bellingham': 253072, 'M. Salah': 209331, 'L. Modrić': 177003,
                'Son Heung-min': 202126, 'A. Griezmann': 194765,
                # ── 新生代 ──
                'K. Havertz': 246669, 'F. Wirtz': 263552, 'L. Yamal': 272456,
                'Pedri': 252371, 'J. David': 254103, 'M. Ødegaard': 231568,
                # ── 中後場明星 ──
                'F. de Jong': 239473, 'F. Valverde': 241764, 'V. van Dijk': 203376,
                'A. Hakimi': 238023, 'J. Kimmich': 214455, 'A. Rüdiger': 206158,
                'A. Robertson': 225321, 'J. Gvardiol': 261255, 'R. Dias': 237692,
                'B. Fernandes': 212831, 'G. Xhaka': 186942, 'M. Akanji': 231678,
                'T. Partey': 209564,
                # ── 進攻群 ──
                'R. Leão': 251706, 'A. Davies': 236460, 'L. Díaz': 243726,
                'S. Mané': 208722, 'R. Mahrez': 220054, 'M. Kudus': 260460,
                'D. Núñez': 261204, 'J. Álvarez': 261289, 'Rodrygo': 252658,
                'D. Kulusevski': 246940,
                # ── 門將 / 後衛 ──
                'M. Neuer': 167495, 'Alisson': 211110, 'Marquinhos': 200389,
            }

            def _sofifa_url(sid: int, version: int = 25, size: int = 120) -> str:
                """SOFIFA CDN 圖片 URL：ID 補滿 6 位數後切 3-3。"""
                s = str(sid).zfill(6)
                return f"https://cdn.sofifa.net/players/{s[:3]}/{s[3:6]}/{version}_{size}.png"

            # 守備位置 中英對照
            POS_NAME_CN = {
                'GK':  '門將',     'CB':  '中後衛',   'LB':  '左後衛',
                'RB':  '右後衛',   'LWB': '左翼衛',   'RWB': '右翼衛',
                'CDM': '防守中場', 'CM':  '中場',     'CAM': '攻擊中場',
                'LM':  '左中場',   'RM':  '右中場',   'LW':  '左翼',
                'RW':  '右翼',     'CF':  '中鋒',     'ST':  '前鋒',
            }
            POS_NAME_EN = {
                'GK':  'Goalkeeper',         'CB':  'Centre-Back',
                'LB':  'Left-Back',          'RB':  'Right-Back',
                'LWB': 'Left Wing-Back',     'RWB': 'Right Wing-Back',
                'CDM': 'Defensive Mid',      'CM':  'Centre Mid',
                'CAM': 'Attacking Mid',
                'LM':  'Left Mid',           'RM':  'Right Mid',
                'LW':  'Left Winger',        'RW':  'Right Winger',
                'CF':  'Centre Forward',     'ST':  'Striker',
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
                        f'<span style="color:#6b6760;font-weight:600;width:28px;">{lbl}</span>'
                        f'<div style="flex:1;background:#f4f1ea;border-radius:4px;height:6px;margin:0 6px;">'
                        f'<div style="width:{val}%;height:6px;border-radius:4px;background:{bc};"></div></div>'
                        f'<span style="color:#1f1e1c;font-weight:600;width:22px;text-align:right;">{val}</span>'
                        f'</div>'
                    )
                # 球員頭像：SOFIFA CDN，未收錄者用姓名縮寫頭像（避免錯誤照片）
                sid = SOFIFA_IDS.get(p['name'], 0)
                initials = ''.join(w[0] for w in p['name'].replace('.','').split() if w)[:2].upper()
                # Claude 配色：高 OVR 珊瑚橘、中 OVR 暖靛、低 OVR 暖灰
                avatar_bg = 'c96442' if ovr >= 85 else ('1f6e8c' if ovr >= 78 else '9b958a')
                fallback_url = (
                    f"https://ui-avatars.com/api/?name={initials}"
                    f"&background={avatar_bg}&color=ffffff&size=120&bold=true&length=2"
                )
                if sid:
                    photo_src_25 = _sofifa_url(sid, 25, 120)
                    photo_src_24 = _sofifa_url(sid, 24, 120)
                    photo_html = (
                        f'<img src="{photo_src_25}" '
                        f'onerror="this.onerror=function(){{this.onerror=null;this.src=\'{fallback_url}\';}};'
                        f'this.src=\'{photo_src_24}\';" '
                        f'style="width:72px;height:72px;border-radius:50%;object-fit:cover;'
                        f'object-position:top;background:#f4f1ea;'
                        f'border:2px solid #c96442;margin-bottom:8px;'
                        f'box-shadow:0 2px 6px rgba(201,100,66,0.18);">'
                    )
                else:
                    photo_html = (
                        f'<img src="{fallback_url}" '
                        f'style="width:72px;height:72px;border-radius:50%;object-fit:cover;'
                        f'border:2px solid #e8e5dd;margin-bottom:8px;'
                        f'box-shadow:0 2px 6px rgba(31,30,28,0.06);">'
                    )
                # Claude 風格 OVR 配色
                ovr_color = '#c96442' if ovr >= 85 else ('#1f6e8c' if ovr >= 78 else '#6b6760')
                # 守備位置中英對照
                pos_code = p["pos"]
                pos_cn = POS_NAME_CN.get(pos_code, pos_code)
                pos_en = POS_NAME_EN.get(pos_code, pos_code)
                cards_html += (
                    f'<div style="background:#ffffff;border:1px solid #e8e5dd;'
                    f'border-radius:12px;padding:16px 14px;text-align:center;'
                    f'box-shadow:0 1px 2px rgba(31,30,28,0.04);">'
                    f'{photo_html}'
                    f'<div style="font-family:Charter,Georgia,serif;font-size:2rem;font-weight:600;'
                    f'color:{ovr_color};line-height:1.1;">{ovr}</div>'
                    # 位置：代碼徽章 + 中文名
                    f'<div style="margin-top:4px;display:flex;justify-content:center;align-items:center;gap:6px;flex-wrap:wrap;">'
                    f'<span style="background:#f3e9e4;color:#c96442;font-size:0.7rem;font-weight:700;'
                    f'padding:2px 8px;border-radius:999px;letter-spacing:0.6px;">{pos_code}</span>'
                    f'<span style="font-size:0.78rem;font-weight:600;color:#1f1e1c;">{pos_cn}</span>'
                    f'</div>'
                    f'<div style="font-size:0.66rem;color:#9b958a;margin-top:2px;font-style:italic;">{pos_en}</div>'
                    f'<hr style="border:none;border-top:1px solid #e8e5dd;margin:8px 0;">'
                    f'<div style="font-size:0.88rem;font-weight:600;color:#1f1e1c;">{p["name"]}</div>'
                    f'<div style="font-size:0.7rem;color:#9b958a;margin-bottom:8px;">{p["club"]} · {p["age"]}歲</div>'
                    f'<hr style="border:none;border-top:1px solid #e8e5dd;margin:8px 0;">'
                    f'{attrs_rows}'
                    f'</div>'
                )

            full_html = (
                f'<div style="display:grid;grid-template-columns:repeat({min(len(players),5)},1fr);gap:12px;">'
                f'{cards_html}</div>'
            )
            _components.html(full_html, height=540, scrolling=False)
            st.caption(
                "📷 球員頭像來源：SOFIFA（FIFA 25 / EAFC 25 資料庫）。"
                "未收錄的球員顯示英文姓名縮寫頭像（黃底=OVR ≥85，青底=78-84，灰底<78）。"
                "若球員照片無法載入會自動 fallback 到縮寫頭像，避免顯示錯誤照片。"
            )

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
            color_discrete_sequence=_CLAUDE_PALETTE,
            height=540,
        )
        fig_pca.update_traces(textposition='top center', marker=dict(size=11),
                              textfont=dict(size=9, color=_CLAUDE_TEXT))
        fig_pca.update_layout(**claude_layout(
            legend_title_text='風格', height=540,
        ))
        st.plotly_chart(fig_pca, use_container_width=True)

        # ── 分群說明：類型定義 + 各群中心數據 ──
        style_colors = {'攻擊型': _CLAUDE_ACCENT, '防守型': _CLAUDE_ACCENT2, '平衡型': '#b58a3b'}
        style_icons  = {'攻擊型': '⚡', '防守型': '🛡️', '平衡型': '⚖️'}
        # 分群依「淨勝球（場均進球−失球）」排序命名，描述與實際數據一致
        style_criteria = {
            '攻擊型': '淨勝球最高（進 2.17 / 失 0.79）→ 進攻火力強、以攻代守',
            '防守型': '淨勝球最低（進 1.14 / 失 1.31）→ 偏防守取向、進攻有限',
            '平衡型': '淨勝球中等（進 1.57 / 失 0.97）→ 攻守均衡、彈性戰術',
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
        st.markdown("### 📡 分群風格雷達圖")
        # 動態繪製：用 48 隊的整體分布做正規化（修正 pretrain.py 用 3 群極端值正規化的 bug）
        import plotly.graph_objects as _go_cr
        radar_feats_cn = ['勝率', '場均進球', '防守強度', '淨勝球', 'FIFA 積分']
        radar_feats_en = ['win_rate', 'avg_goals', 'avg_conceded', 'goal_diff', 'fifa_pts']
        # 用整體 48 隊（df_c）做 min-max 而非僅 3 個 cluster center
        # 注意：avg_conceded 越低越好，所以反向（1 - normalized）
        feat_min = df_c[radar_feats_en].min()
        feat_max = df_c[radar_feats_en].max()
        center_data = centers[radar_feats_en].copy()
        center_norm = (center_data - feat_min) / (feat_max - feat_min + 1e-9)
        center_norm['avg_conceded'] = 1 - center_norm['avg_conceded']  # 反向：低失球 → 高分

        cluster_palette = {'攻擊型': '#c96442', '防守型': '#1f6e8c', '平衡型': '#b58a3b'}
        fig_clu_radar = _go_cr.Figure()
        for cid in sorted(cluster_names.keys()):
            cn_raw = cluster_names[cid]
            cn_pure = cn_raw.replace('⚡', '').replace('🛡️', '').replace('⚖️', '').strip()
            color = cluster_palette.get(cn_pure, _CLAUDE_ACCENT)
            vals = center_norm.iloc[cid].tolist()
            fig_clu_radar.add_trace(_go_cr.Scatterpolar(
                r=vals + [vals[0]],
                theta=radar_feats_cn + [radar_feats_cn[0]],
                fill='toself',
                name=cn_pure,
                line=dict(color=color, width=2.5),
                fillcolor=color.replace('#', 'rgba(') if False else _hex_to_rgba(color, 0.15),
            ))
        fig_clu_radar.update_layout(**claude_layout(
            polar=dict(
                bgcolor=_CLAUDE_CARD,
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor=_CLAUDE_GRID, color=_CLAUDE_TEXT,
                                tickfont=dict(size=10)),
                angularaxis=dict(gridcolor=_CLAUDE_GRID, color=_CLAUDE_TEXT,
                                 tickfont=dict(size=12)),
            ),
            showlegend=True, height=480,
            legend=dict(bgcolor='rgba(255,255,255,0.85)',
                        bordercolor=_CLAUDE_GRID, borderwidth=1),
            title=dict(text=f'三類風格中心雷達（k={k}, Silhouette={sil:.2f}）',
                       font=dict(family='Charter, Georgia, serif',
                                 color=_CLAUDE_TEXT, size=15)),
        ))
        st.plotly_chart(fig_clu_radar, use_container_width=True)
        st.caption(
            "📖 **怎麼讀**：5 個維度都用 **48 隊整體分布** 做 0-1 正規化（非僅 3 群極值）。"
            "離中心越遠 = 該指標越強；「防守強度」已反向，越外圈代表越少失球。"
        )

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
                line_color=_CLAUDE_ACCENT, fillcolor='rgba(201,100,66,0.15)',
            ))
            fig_radar.add_trace(_go_r.Scatterpolar(
                r=norm_b + [norm_b[0]], theta=radar_labels + [radar_labels[0]],
                fill='toself', name=TEAM_INFO.get(team_b, {'cn': team_b})['cn'],
                line_color=_CLAUDE_ACCENT2, fillcolor='rgba(31,110,140,0.15)',
            ))
            fig_radar.update_layout(**claude_layout(
                polar=dict(
                    bgcolor=_CLAUDE_CARD,
                    radialaxis=dict(visible=True, range=[0, 1],
                                    gridcolor=_CLAUDE_GRID, color=_CLAUDE_TEXT),
                    angularaxis=dict(gridcolor=_CLAUDE_GRID, color=_CLAUDE_TEXT),
                ),
                showlegend=True, height=450,
                title=dict(text='兩隊攻守風格雷達比較（正規化）',
                           font=dict(family='Charter, Georgia, serif',
                                     color=_CLAUDE_TEXT, size=15)),
            ))
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
        _mc_xmax = float(top20['champ_pct'].max()) * 1.25
        fig_mc.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis=dict(range=[0, _mc_xmax], ticksuffix='%'),
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

        # ════════════════════════════════════════════════════════
        # 市場校準層（Layer 5）：模型 ⊕ 博彩市場共識
        # ════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 🎰 市場校準：模型 ⊕ 博彩賠率")

        # 時間衰減權重：離開賽越近，市場越準 → α 從 0.20 漸增到 0.50
        try:
            _start = _date.fromisoformat(MARKET.TOURNAMENT_START)
            _days = (_start - _date.today()).days
        except Exception:
            _days = 0
        alpha = max(0.20, min(0.50, 0.50 - 0.30 * max(_days, 0) / 120.0))

        # 模型機率（champ_pct/100，總和=1）vs 市場去抽水機率
        model_p = {r['team']: r['champ_pct'] / 100.0 for _, r in mc_df.iterrows()}
        market_p = MARKET.market_implied_probs()
        all_t = set(model_p) | set(market_p)
        blended = {t: (1 - alpha) * model_p.get(t, 0.0) + alpha * market_p.get(t, 0.0)
                   for t in all_t}
        _bs = sum(blended.values()) or 1.0
        blended = {t: v / _bs for t, v in blended.items()}

        # 時間衰減說明卡
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.metric("距開賽天數", f"{max(_days,0)} 天",
                      help=f"開賽日 {MARKET.TOURNAMENT_START}")
        with cc2:
            st.metric("市場權重 α", f"{alpha:.0%}",
                      delta="越近賽事越高", delta_color="off",
                      help=("α = max(0.20, min(0.50, 0.50 − 0.30 × 距開賽天數 / 120))。"
                            f"參考開賽日 {MARKET.TOURNAMENT_START}；"
                            "距賽 120 天以上鎖定 20%，開賽日達上限 50%，中間線性內插。"))
        with cc3:
            st.metric("莊家抽水", f"{MARKET.market_overround()*100:.0f}%",
                      help=f"賠率快照 {MARKET.AS_OF}；已用 de-vig 去除")

        st.caption(
            f"📐 **融合公式**：最終 = (1−α)×模型 + α×市場　|　目前 "
            f"**{1-alpha:.0%} 模型 + {alpha:.0%} 市場**（α 隨開賽逼近上升）。"
            f"市場機率來自運彩冠軍盤口賠率去除莊家抽水（快照 {MARKET.AS_OF}，每日自動更新）。"
        )

        # 三欄對照表：模型 / 市場 / 融合（依融合排序）
        cmp_rows = []
        for t in sorted(blended, key=lambda x: -blended[x])[:15]:
            info = TEAM_INFO.get(t, {'flag': '🏳️', 'cn': t})
            cmp_rows.append({
                '球隊': f"{info['flag']} {info['cn']}",
                '🤖 模型': f"{model_p.get(t,0)*100:.1f}%",
                '🎰 市場': f"{market_p.get(t,0)*100:.1f}%",
                '⊕ 融合': f"{blended[t]*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

        # 融合 vs 模型 vs 市場 分組長條圖
        import plotly.graph_objects as _go_mk
        topn = sorted(blended, key=lambda x: -blended[x])[:10]
        labels = [TEAM_INFO.get(t, {'cn': t})['cn'] for t in topn]
        fig_mk = _go_mk.Figure()
        fig_mk.add_trace(_go_mk.Bar(name='🤖 模型', x=labels,
                                    y=[model_p.get(t, 0)*100 for t in topn],
                                    marker_color=_CLAUDE_ACCENT2))
        fig_mk.add_trace(_go_mk.Bar(name='🎰 市場', x=labels,
                                    y=[market_p.get(t, 0)*100 for t in topn],
                                    marker_color='#b58a3b'))
        fig_mk.add_trace(_go_mk.Bar(name='⊕ 融合', x=labels,
                                    y=[blended[t]*100 for t in topn],
                                    marker_color=_CLAUDE_ACCENT))
        fig_mk.update_layout(**claude_layout(
            barmode='group', height=420,
            yaxis_title='奪冠機率 (%)',
            legend=dict(orientation='h', y=1.12,
                        bgcolor='rgba(255,255,255,0.85)',
                        bordercolor=_CLAUDE_GRID, borderwidth=1),
        ))
        st.plotly_chart(fig_mk, use_container_width=True)

        with st.expander("📖 詳細解讀（市場校準在做什麼？）", expanded=False):
            st.markdown(
                f"""
**🎯 為什麼要校準？**

我們的 ML 模型是「**歷史結構模型**」—— 重 FIFA 積分、攻防數據、陣容 OVR。
但它**看不到**：傷病、即時俱樂部狀態、戰術成熟度、抽籤路徑這些「當下」資訊。

博彩市場用**真金白銀**把這些難量化的因素濃縮成一個數字（賠率）。
學術上博彩市場被視為**最強的單一預測基準**（efficient market）。

**📐 怎麼量化市場？（純數據，無主觀）**

1. 十進位賠率 → 隱含機率 = 1 / 賠率
2. 加總得 overround（含莊家抽水 {MARKET.market_overround()*100:.0f}%）
3. 除以 overround → 去抽水的「真實市場機率」（總和 = 1）

**🔄 兩層浮動設計**

- **賠率浮動**：`market_odds.py` 每天凌晨由 GitHub Action 自動重抓更新
- **權重浮動**：α 隨開賽逼近從 20% 升到 50%
  （離賽越近，市場資訊越完整越準 → 該更信市場）
  目前距開賽 {max(_days,0)} 天 → α = {alpha:.0%}

**💡 結果怎麼看**

- 🤖 模型：純歷史結構觀點（巴西/阿根廷因 FIFA 積分領先）
- 🎰 市場：當下共識（西班牙/法國因近期狀態+陣容深度領先）
- ⊕ 融合：兩者加權 —— 兼顧「長期實力」與「當下狀態」

這不是用市場「作弊」，而是把「市場情緒」當成一個**可量化、會浮動的特徵**納入，
正是你要的「以數據推論 + 隨時間浮動」。
"""
            )

# ============================================================
# PAGE 6: 完整賽程
# ============================================================
elif page == "📅 完整賽程":
    st.title("📅 2026 世界盃完整賽程")
    st.caption("全程台灣時間（UTC+8）· 小組賽 6/12–6/28（72 場，依日期排列）· 淘汰賽 6/29–7/20 · 資料依 FIFA 2026 官方賽程")

    import streamlit.components.v1 as _c
    # ════════ 小組賽：依日期排列 ＋ 模型預測比分 ＋ ESPN 即時實際比分 ════════
    _sc_l, _sc_r = st.columns([3, 1])
    with _sc_l:
        st.subheader("🗓️ 小組賽賽程（依日期）")
    with _sc_r:
        if st.button("🔄 立即更新比分", use_container_width=True):
            fetch_wc_live_scores.clear()
    st.caption("每列：開球時間 · 組別 · 客隊 vs 主隊（主隊置右）·〔預〕最可能比分＋預測勝率"
               "＋和局率（藍＝勝率、金＝和局率；模型針對勝負最佳化，精確比分僅示意）·〔實〕實際比分（金；完賽後比對「✓命中＝勝負方向正確」）。"
               "進行中顯示 🔴 LIVE，比分格式為「客-主」。資料來源：ESPN，每 45 秒自動刷新。")
    if hasattr(st, "fragment"):
        st.fragment(render_group_stage_schedule, run_every=45)()
    else:
        render_group_stage_schedule()

    st.markdown("---")
    st.subheader("🏆 淘汰賽對陣圖")
    st.caption("🏆決賽（頂端）向下展開至 32 強 · 日期依 FIFA 官方賽程（R32 6/29 → 決賽 7/20，台灣時間）· ⚠️ 對陣組合為示意，以官方抽籤後實際晉級為準")

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
    # 日期依 EBC/FIFA 2026 官方賽程（台灣時間）：R32 6/29-7/4 · R16 7/5-7/8 · QF 7/10-7/12 · SF 7/15-7/16 · 決賽 7/20
    r32_all = [
        ("6/29 03:00","A組1","B組2"), ("6/29 07:00","C組1","D組2"),
        ("6/30 01:00","E組1","F組2"), ("6/30 09:00","G組1","H組2"),
        ("7/1 01:00","I組1","J組2"), ("7/1 09:00","K組1","L組2"),
        ("7/2 00:00","最佳3rd①","最佳3rd②"), ("7/2 08:00","最佳3rd③","最佳3rd④"),
        ("6/30 04:30","B組1","A組2"), ("7/1 05:00","D組1","C組2"),
        ("7/2 04:00","F組1","E組2"), ("7/3 03:00","H組1","G組2"),
        ("7/3 07:00","J組1","I組2"), ("7/3 11:00","L組1","K組2"),
        ("7/4 02:00","最佳3rd⑤","最佳3rd⑥"), ("7/4 06:00","最佳3rd⑦","最佳3rd⑧"),
    ]
    r16_all = [
        ("7/5 01:00","R32①勝","R32②勝"), ("7/5 05:00","R32③勝","R32④勝"),
        ("7/6 04:00","R32⑤勝","R32⑥勝"), ("7/6 08:00","R32⑦勝","R32⑧勝"),
        ("7/7 03:00","R32⑨勝","R32⑩勝"), ("7/7 08:00","R32⑪勝","R32⑫勝"),
        ("7/8 00:00","R32⑬勝","R32⑭勝"), ("7/8 04:00","R32⑮勝","R32⑯勝"),
    ]
    qf_all = [
        ("7/10 04:00","16強①勝","16強②勝"), ("7/11 03:00","16強③勝","16強④勝"),
        ("7/12 05:00","16強⑤勝","16強⑥勝"), ("7/12 09:00","16強⑦勝","16強⑧勝"),
    ]
    sf_all = [
        ("7/15 03:00","八強①勝","八強②勝"),
        ("7/16 03:00","八強③勝","八強④勝"),
    ]
    fin_d = ("7/20 03:00", "四強①勝", "四強②勝")

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
        f'7/19 · 四強負方① VS 四強負方②</span>'
        f'</div>'
    )

    full_html = (
        f'<div style="background:#091525;border-radius:12px;padding:10px 12px;'
        f'font-family:\'Noto Sans TC\',sans-serif;overflow-x:auto;min-width:{TW}px;">'
        f'{body}{third_bar}'
        f'<div style="text-align:center;color:#1a3050;font-size:0.55rem;margin-top:5px;">'
        f'※ 日期依 EBC/FIFA 2026 官方賽程（6/29-7/20，台灣時間 UTC+8）· 對陣組合以官方抽籤為準</div>'
        f'</div>'
    )

    st.caption(f"賽程總寬度 {TW}px · 若未完整顯示可左右滑動")
    _c.html(full_html, height=CANVAS_H + 85, scrolling=True)


# ════════════════════════════════════════════════════════
# 全域頁尾：學術用途免責聲明（每頁底部顯示）
# ════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "⚠️ **學術用途免責聲明**：本系統為逢甲大學數據科學課程期末專題，所有預測由機器學習模型依公開歷史資料"
    "自動產生，僅供學術研究與技術展示，**不構成任何投注、博弈或財務建議**，亦不保證準確。詳見下方完整聲明。"
)
with st.expander("📜 完整免責聲明 / Full Disclaimer"):
    st.markdown(
        """
**本專案為學術研究與教學用途，不構成任何投注或財務建議。**

1. **學術用途**：本系統（2026 世界盃 ML 預測系統）係逢甲大學「數據科學（碩士班）」課程之期末專題，純為學術研究、技術學習與教學示範，屬非營利、非商業性質。
2. **非投注／賭博建議**：所有預測（比分、勝負、奪冠／晉級機率等）皆由機器學習模型依公開歷史資料自動計算，僅供學術參考，**不構成、也不應被視為任何形式之投注、博弈、財務或投資建議**，亦非任何下注行為之推薦、邀約或保證。
3. **不保證準確性**：足球賽事具高度不確定性，模型輸出為機率性估計，無法保證準確；歷史表現不代表未來結果。任何依本系統資訊所做之決策，須自行承擔全部風險，作者與相關單位概不負責。
4. **市場賠率之引用**：「市場校準」僅將公開博彩賠率作為模型校準與學術比較之資料，**不代表推薦、背書或鼓勵任何博彩業者、平台或下注行為**。
5. **合法與責任博弈**：請遵守您所在地區之法律規定。本專案不鼓勵任何賭博行為；若您或身邊的人受賭博問題困擾，請尋求專業協助（如台灣 1995 安心專線）。
6. **資料來源**：歷史賽事 [martj42/international_results](https://github.com/martj42/international_results)、FIFA 排名 [Dato-Futbol/fifa-ranking](https://github.com/Dato-Futbol/fifa-ranking)，著作權歸原作者，本專案僅作非商業學術使用。

*English — This is an academic course project for educational/research purposes only. All predictions are machine-generated probabilistic estimates and do NOT constitute betting, gambling, financial, or investment advice, nor any endorsement or solicitation to wager. No accuracy is guaranteed — use at your own risk. Please gamble responsibly and comply with local laws.*
        """
    )

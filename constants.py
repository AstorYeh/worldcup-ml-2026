"""
constants.py — 2026 World Cup ML 專案的單一資料來源 (Single Source of Truth)
============================================================================
app.py 與 pretrain.py 都從這裡 import，避免資料重複定義導致漂移。

包含：
    TEAM_INFO       48 隊：國旗 / ISO / 中英文名 / FIFA 排名 / FIFA 積分
    WC_2026_GROUPS  12 組分組
    CONFED_MAP      48 隊所屬足聯（完整，無缺漏）
    CONFED_BONUS    各足聯歷史勝率加成
    TEAM_PTS        從 TEAM_INFO 衍生：{隊名: fifa_pts}
    TEAM_RANK       從 TEAM_INFO 衍生：{隊名: fifa_rank}

⚠️ 修改球隊資料只改這個檔案，兩支程式自動同步。
"""

# ============================================================
# 48 支球隊：國旗 + 中英文名 + FIFA 排名/積分（2026/4 FIFA 官方）
# ============================================================
TEAM_INFO = {
    # FIFA 世界排名（2026年4月 / 資料來源：FIFA官網）
    # === GROUP A ===
    'Mexico':         {'flag': '🇲🇽', 'iso': 'mx', 'cn': '墨西哥',         'en': 'Mexico',          'fifa_rank': 15, 'fifa_pts': 1685},
    'Czechia':        {'flag': '🇨🇿', 'iso': 'cz', 'cn': '捷克',           'en': 'Czechia',         'fifa_rank': 43, 'fifa_pts': 1440},
    'South Korea':    {'flag': '🇰🇷', 'iso': 'kr', 'cn': '南韓',           'en': 'South Korea',     'fifa_rank': 24, 'fifa_pts': 1569},
    'South Africa':   {'flag': '🇿🇦', 'iso': 'za', 'cn': '南非',           'en': 'South Africa',    'fifa_rank': 56, 'fifa_pts': 1396},
    # === GROUP B ===
    'Canada':         {'flag': '🇨🇦', 'iso': 'ca', 'cn': '加拿大',         'en': 'Canada',          'fifa_rank': 29, 'fifa_pts': 1530},
    'Switzerland':    {'flag': '🇨🇭', 'iso': 'ch', 'cn': '瑞士',           'en': 'Switzerland',     'fifa_rank': 19, 'fifa_pts': 1632},
    'Qatar':          {'flag': '🇶🇦', 'iso': 'qa', 'cn': '卡達',           'en': 'Qatar',           'fifa_rank': 53, 'fifa_pts': 1413},
    'Bosnia and Herzegovina': {'flag': '🇧🇦', 'iso': 'ba', 'cn': '波赫',   'en': 'Bosnia',          'fifa_rank': 78, 'fifa_pts': 1325},
    # === GROUP C ===
    'Brazil':         {'flag': '🇧🇷', 'iso': 'br', 'cn': '巴西',           'en': 'Brazil',          'fifa_rank': 5,  'fifa_pts': 1819},
    'Morocco':        {'flag': '🇲🇦', 'iso': 'ma', 'cn': '摩洛哥',         'en': 'Morocco',         'fifa_rank': 13, 'fifa_pts': 1695},
    'Scotland':       {'flag': '🏴󠁧󠁢󠁳󠁣󠁴󠁿', 'iso': 'gb-sct', 'cn': '蘇格蘭', 'en': 'Scotland',      'fifa_rank': 38, 'fifa_pts': 1480},
    'Haiti':          {'flag': '🇭🇹', 'iso': 'ht', 'cn': '海地',           'en': 'Haiti',           'fifa_rank': 86, 'fifa_pts': 1279},
    # === GROUP D ===
    'USA':            {'flag': '🇺🇸', 'iso': 'us', 'cn': '美國',           'en': 'USA',             'fifa_rank': 17, 'fifa_pts': 1635},
    'Paraguay':       {'flag': '🇵🇾', 'iso': 'py', 'cn': '巴拉圭',         'en': 'Paraguay',        'fifa_rank': 40, 'fifa_pts': 1475},
    'Australia':      {'flag': '🇦🇺', 'iso': 'au', 'cn': '澳洲',           'en': 'Australia',       'fifa_rank': 26, 'fifa_pts': 1544},
    'Turkiye':        {'flag': '🇹🇷', 'iso': 'tr', 'cn': '土耳其',         'en': 'Turkiye',         'fifa_rank': 27, 'fifa_pts': 1540},
    # === GROUP E ===
    'Germany':        {'flag': '🇩🇪', 'iso': 'de', 'cn': '德國',           'en': 'Germany',         'fifa_rank': 14, 'fifa_pts': 1692},
    'Ecuador':        {'flag': '🇪🇨', 'iso': 'ec', 'cn': '厄瓜多',         'en': 'Ecuador',         'fifa_rank': 28, 'fifa_pts': 1535},
    'Ivory Coast':    {'flag': '🇨🇮', 'iso': 'ci', 'cn': '象牙海岸',       'en': 'Ivory Coast',     'fifa_rank': 42, 'fifa_pts': 1448},
    'Curacao':        {'flag': '🇨🇼', 'iso': 'cw', 'cn': '庫拉索',         'en': 'Curacao',         'fifa_rank': 79, 'fifa_pts': 1293},
    # === GROUP F ===
    'Netherlands':    {'flag': '🇳🇱', 'iso': 'nl', 'cn': '荷蘭',           'en': 'Netherlands',     'fifa_rank': 7,  'fifa_pts': 1760},
    'Japan':          {'flag': '🇯🇵', 'iso': 'jp', 'cn': '日本',           'en': 'Japan',           'fifa_rank': 16, 'fifa_pts': 1640},
    'Tunisia':        {'flag': '🇹🇳', 'iso': 'tn', 'cn': '突尼斯',         'en': 'Tunisia',         'fifa_rank': 49, 'fifa_pts': 1429},
    'Sweden':         {'flag': '🇸🇪', 'iso': 'se', 'cn': '瑞典',           'en': 'Sweden',          'fifa_rank': 39, 'fifa_pts': 1478},
    # === GROUP G ===
    'Belgium':        {'flag': '🇧🇪', 'iso': 'be', 'cn': '比利時',         'en': 'Belgium',         'fifa_rank': 6,  'fifa_pts': 1768},
    'Iran':           {'flag': '🇮🇷', 'iso': 'ir', 'cn': '伊朗',           'en': 'Iran',            'fifa_rank': 20, 'fifa_pts': 1622},
    'Egypt':          {'flag': '🇪🇬', 'iso': 'eg', 'cn': '埃及',           'en': 'Egypt',           'fifa_rank': 33, 'fifa_pts': 1516},
    'New Zealand':    {'flag': '🇳🇿', 'iso': 'nz', 'cn': '紐西蘭',         'en': 'New Zealand',     'fifa_rank': 88, 'fifa_pts': 1270},
    # === GROUP H ===
    'Spain':          {'flag': '🇪🇸', 'iso': 'es', 'cn': '西班牙',         'en': 'Spain',           'fifa_rank': 2,  'fifa_pts': 1875},
    'Uruguay':        {'flag': '🇺🇾', 'iso': 'uy', 'cn': '烏拉圭',         'en': 'Uruguay',         'fifa_rank': 11, 'fifa_pts': 1701},
    'Saudi Arabia':   {'flag': '🇸🇦', 'iso': 'sa', 'cn': '沙烏地阿拉伯',   'en': 'Saudi Arabia',    'fifa_rank': 58, 'fifa_pts': 1390},
    'Cape Verde':     {'flag': '🇨🇻', 'iso': 'cv', 'cn': '維德角',         'en': 'Cape Verde',      'fifa_rank': 73, 'fifa_pts': 1340},
    # === GROUP I ===
    'France':         {'flag': '🇫🇷', 'iso': 'fr', 'cn': '法國',           'en': 'France',          'fifa_rank': 3,  'fifa_pts': 1870},
    'Senegal':        {'flag': '🇸🇳', 'iso': 'sn', 'cn': '塞內加爾',       'en': 'Senegal',         'fifa_rank': 21, 'fifa_pts': 1621},
    'Norway':         {'flag': '🇳🇴', 'iso': 'no', 'cn': '挪威',           'en': 'Norway',          'fifa_rank': 30, 'fifa_pts': 1525},
    'Iraq':           {'flag': '🇮🇶', 'iso': 'iq', 'cn': '伊拉克',         'en': 'Iraq',            'fifa_rank': 55, 'fifa_pts': 1405},
    # === GROUP J ===
    'Argentina':      {'flag': '🇦🇷', 'iso': 'ar', 'cn': '阿根廷',         'en': 'Argentina',       'fifa_rank': 1,  'fifa_pts': 1886},
    'Austria':        {'flag': '🇦🇹', 'iso': 'at', 'cn': '奧地利',         'en': 'Austria',         'fifa_rank': 22, 'fifa_pts': 1580},
    'Algeria':        {'flag': '🇩🇿', 'iso': 'dz', 'cn': '阿爾及利亞',     'en': 'Algeria',         'fifa_rank': 31, 'fifa_pts': 1522},
    'Jordan':         {'flag': '🇯🇴', 'iso': 'jo', 'cn': '約旦',           'en': 'Jordan',          'fifa_rank': 68, 'fifa_pts': 1356},
    # === GROUP K ===
    'Portugal':       {'flag': '🇵🇹', 'iso': 'pt', 'cn': '葡萄牙',         'en': 'Portugal',        'fifa_rank': 8,  'fifa_pts': 1752},
    'Colombia':       {'flag': '🇨🇴', 'iso': 'co', 'cn': '哥倫比亞',       'en': 'Colombia',        'fifa_rank': 9,  'fifa_pts': 1739},
    'Uzbekistan':     {'flag': '🇺🇿', 'iso': 'uz', 'cn': '烏茲別克',       'en': 'Uzbekistan',      'fifa_rank': 54, 'fifa_pts': 1410},
    'DR Congo':       {'flag': '🇨🇩', 'iso': 'cd', 'cn': '民主剛果',       'en': 'DR Congo',        'fifa_rank': 61, 'fifa_pts': 1359},
    # === GROUP L ===
    'England':        {'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'iso': 'gb-eng', 'cn': '英格蘭', 'en': 'England',      'fifa_rank': 4,  'fifa_pts': 1825},
    'Croatia':        {'flag': '🇭🇷', 'iso': 'hr', 'cn': '克羅埃西亞',     'en': 'Croatia',         'fifa_rank': 12, 'fifa_pts': 1700},
    'Panama':         {'flag': '🇵🇦', 'iso': 'pa', 'cn': '巴拿馬',         'en': 'Panama',          'fifa_rank': 37, 'fifa_pts': 1503},
    'Ghana':          {'flag': '🇬🇭', 'iso': 'gh', 'cn': '迦納',           'en': 'Ghana',           'fifa_rank': 70, 'fifa_pts': 1353},
}

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
# 足聯歸屬與加成
# ============================================================
# 足聯歷史強度加成（XGBoost confed_bonus 特徵用）
# v2: CONMEBOL 0.10→0.06 — 現代歐洲整體深度已追平南美，原值過度獎勵巴西/阿根廷晉級
CONFED_BONUS = {
    'UEFA': 0.08, 'CONMEBOL': 0.06, 'CONCACAF': 0.02,
    'AFC': 0.01, 'CAF': 0.00, 'OFC': 0.00
}

CONFED_MAP = {
    # UEFA
    'France': 'UEFA', 'Spain': 'UEFA', 'England': 'UEFA', 'Germany': 'UEFA',
    'Portugal': 'UEFA', 'Netherlands': 'UEFA', 'Belgium': 'UEFA',
    'Croatia': 'UEFA', 'Switzerland': 'UEFA', 'Sweden': 'UEFA',
    'Austria': 'UEFA', 'Czechia': 'UEFA', 'Scotland': 'UEFA', 'Norway': 'UEFA',
    'Bosnia and Herzegovina': 'UEFA', 'Turkiye': 'UEFA',
    # CONMEBOL
    'Brazil': 'CONMEBOL', 'Argentina': 'CONMEBOL', 'Uruguay': 'CONMEBOL',
    'Colombia': 'CONMEBOL', 'Ecuador': 'CONMEBOL', 'Paraguay': 'CONMEBOL',
    # CONCACAF
    'USA': 'CONCACAF', 'Mexico': 'CONCACAF', 'Canada': 'CONCACAF',
    'Panama': 'CONCACAF', 'Curacao': 'CONCACAF', 'Haiti': 'CONCACAF',
    # AFC
    'Japan': 'AFC', 'South Korea': 'AFC', 'Iran': 'AFC', 'Australia': 'AFC',
    'Saudi Arabia': 'AFC', 'Iraq': 'AFC', 'Jordan': 'AFC', 'Uzbekistan': 'AFC',
    'Qatar': 'AFC',
    # OFC
    'New Zealand': 'OFC',
    # CAF
    'Morocco': 'CAF', 'Egypt': 'CAF', 'Senegal': 'CAF', 'Algeria': 'CAF',
    'Ghana': 'CAF', 'Ivory Coast': 'CAF', 'Tunisia': 'CAF', 'Cape Verde': 'CAF',
    'DR Congo': 'CAF', 'South Africa': 'CAF',
}


# ============================================================
# 從 TEAM_INFO 衍生（pretrain.py 訓練特徵使用）
# ============================================================
TEAM_PTS = {team: info['fifa_pts'] for team, info in TEAM_INFO.items()}
TEAM_RANK = {team: info['fifa_rank'] for team, info in TEAM_INFO.items()}

# 所有 48 隊扁平列表
ALL_TEAMS = [t for teams in WC_2026_GROUPS.values() for t in teams]

# 🏆 2026 World Cup ML Prediction System

> 數據科學期末專題 — 用機器學習預測 2026 FIFA 世界盃比賽結果與冠軍

[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-success?logo=streamlit)](https://worldcup-ml-2026-bzplzw7hoy7g5dcizaakvt.streamlit.app/)

🌐 **線上互動 Demo**：<https://worldcup-ml-2026-bzplzw7hoy7g5dcizaakvt.streamlit.app/>

---

## 📊 專題概覽

| 項目 | 內容 |
|------|------|
| **資料規模** | 49,257 場國際比賽（1872-2026）+ 67,894 筆 FIFA 排名 + 240 球員陣容 |
| **分類模型** | XGBoost（勝/平/負三向）· Walk-Forward 4 屆 WC 驗證 |
| **迴歸模型** | Dixon-Coles Poisson（主客隊期望進球 λ） |
| **分群模型** | KMeans + PCA · 48 支球隊風格分群（攻擊/平衡/防守） |
| **模擬** | Monte Carlo 10,000 次完整賽程奪冠模擬 |
| **驗證準確率** | 52.0%（vs 隨機猜測 33.3%，提升 +18.7%） |

---

## 🚀 快速啟動

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 一次性訓練（產出 models/*.pkl，約 1-2 分鐘）
python pretrain.py

# 3. 啟動 Streamlit
streamlit run app.py
```

> ⚡ **執行 `pretrain.py` 是必要的**：所有 ML 模型（分類/迴歸/分群/Monte Carlo）都已預先訓練並存成 `models/*.pkl`，
> 開啟 Streamlit 時直接讀檔，**徹底消除 Streamlit Cloud 冷啟動 30 秒等待**。

---

## 🧠 五層融合模型架構（含市場校準）

```
┌──────────────────────────────────────────────────────┐
│ L1  XGBoost 分類器               [外層權重 20%]    │
│     輸出 P(勝/平/負)                                  │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│ L2  Dixon-Coles Poisson          [外層權重 80%]    │
│     λ_A = atk_A × vul_B / μ                          │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│ L3  λ 多因素修正                 [乘法修正]         │
│     主將 OVR ^0.35 · FIFA ^0.22 · 狀態 ^0.18 · 經驗 ^0.08 │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│ L4  融合機率 + Monte Carlo       [賽程模擬]         │
│     P = 0.20 × XGB + 0.80 × Poisson + MC × 10K     │
│     （MC 含主將陣容 OVR 修正）                       │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│ L5  市場校準（奪冠頁）           [時間浮動]         │
│     最終 = (1−α)×模型 + α×市場                       │
│     α 隨開賽逼近 0.20→0.50；市場=賠率去抽水隱含機率  │
│     賠率每日由 Docker 容器自動更新                   │
└──────────────────────────────────────────────────────┘
```

> **市場校準（L5）**：把博彩冠軍盤口賠率去除莊家抽水（de-vig）轉成「市場共識機率」，
> 與模型加權融合。權重 α 隨開賽逼近上升（離賽越近市場越準）。
> 賠率快照 `market_odds.py` 由 **Docker 容器 `docker/odds-updater/`** 每日凌晨自動重抓並 git push 更新
> （在 `.env` 設 `ODDS_API_KEY` + `GITHUB_TOKEN`；未設定則沿用快照，app 照常運作）。
> 詳見 [`docker/odds-updater/README.md`](docker/odds-updater/README.md)。

完整公式與細節請見 [Streamlit App](https://worldcup-ml-2026-bzplzw7hoy7g5dcizaakvt.streamlit.app/) 「📊 專題總覽 → 🔬 模型架構」。

---

## 📁 專案結構

```
worldcup-ml-2026/
├── app.py                  # Streamlit 主程式（7 個分頁）
├── pretrain.py             # 離線訓練腳本（必跑一次）
├── squad_data.py           # 48 隊 × 5 主將陣容資料（240 球員）
├── requirements.txt        # Python 依賴
├── README.md               # 本檔案
│
├── .streamlit/
│   └── config.toml         # Streamlit 主題設定（Claude 暖色）
│
├── assets/
│   ├── hero.jpg            # 主頁 hero banner
│   └── logo.png            # 側邊欄 LOGO
│
├── models/                 # 預訓練模型（pretrain.py 產出）
│   ├── clf.pkl             # XGBoost 分類器
│   ├── poisson1.pkl        # Team1 進球 Poisson
│   ├── poisson2.pkl        # Team2 進球 Poisson
│   ├── feat_cols.pkl       # 特徵欄位列表
│   ├── val_accs.pkl        # Walk-Forward 4 屆 WC 驗證準確率
│   ├── mc_results.pkl      # Monte Carlo 10,000 次奪冠機率
│   ├── team_clusters.pkl   # 球隊風格分群（K-Means + PCA）
│   └── eval_metrics.pkl    # 混淆矩陣 / ROC / 校準 / 特徵重要性
│
└── presentation/
    └── build_ppt.js        # 期末報告 PPT 生成腳本（pptxgenjs）
```

---

## 📱 應用介面（7 個分頁）

| 分頁 | 內容 |
|------|------|
| 📊 **專題總覽** | Hero 圖、資料規模、12 組分組卡、五層架構（4 層模型 + 市場校準，垂直流程圖） |
| 🔮 **2026 預測** | 各組逐場比分預測、Walk-Forward 驗證結果、詳細分析（雷達+熱圖+H2H+TOP5 主將） |
| 📈 **數據分析** | Poisson 分佈驗證、進球趨勢、死亡之組分析、5 個模型評估 Tab（混淆矩陣/ROC/校準/特徵/費雪檢定） |
| 🌍 **各國分析** | 球隊歷史數據、近期賽果、5 位主將能力卡（含位置中英對照 + SOFIFA 頭像） |
| 🎯 **球隊風格分群** | PCA 二維散點、三類風格雷達圖、兩隊風格比較 |
| 🏅 **奪冠預測** | Monte Carlo Top 20、進決賽/小組第一機率、**市場校準層**（模型⊕博彩賠率，權重隨開賽浮動）|
| 📅 **完整賽程** | 32 強淘汰賽 bracket 圖 |

---

## 🎯 對應課程要求的三大 ML 任務

1. **分類** — XGBoost 三向勝負預測 · Walk-Forward 跨 4 屆世界盃驗證
2. **迴歸** — Dixon-Coles Poisson 預測進球數（主客分開模型 + λ 多因素修正）
3. **分群** — KMeans + PCA 對 48 隊做攻守風格非監督分群（k=3, silhouette=0.33）

---

## 🔬 模型驗證結果

### Walk-Forward 滾動驗證

| 訓練資料 | 驗證集 | 準確率 |
|---------|--------|--------|
| 1990-2009 | WC 2010 | 53.3% |
| 1990-2013 | WC 2014 | 48.4% |
| 1990-2017 | WC 2018 | 50.0% |
| 1990-2021 | WC 2022 | 56.2% |
| **平均** | — | **52.0%** |

→ 比隨機猜測 33.3% 高出 **+18.7 個百分點**

### 費雪精確檢定（α=0.05）

| 類別 | p-value | Odds Ratio | 結論 |
|------|---------|-----------|------|
| 主隊勝 | < 0.001 | 8.94 | ✅ 極顯著 |
| 主隊負 | 0.0014 | 5.20 | ✅ 極顯著 |
| 平局 | 0.6234 | 1.08 | ⚠️ 未達顯著（業界共同難題） |

### Monte Carlo Top 5 奪冠機率（10,000 次模擬）

| 排名 | 球隊 | 奪冠機率 |
|------|------|---------|
| 🥇 | 🇦🇷 阿根廷 | 6.22% |
| 🥈 | 🇧🇷 巴西 | 6.03% |
| 🥉 | 🇫🇷 法國 | 5.65% |
| 4 | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 英格蘭 | 5.49% |
| 5 | 🇪🇸 西班牙 | 5.01% |

---

## 🛠️ 技術棧

- **資料處理**：pandas, numpy
- **機器學習**：xgboost, scikit-learn (PoissonRegressor, KMeans, PCA)
- **統計檢定**：scipy.stats (Fisher's Exact, Chi-square)
- **視覺化**：Plotly (interactive), matplotlib (fallback)
- **網頁框架**：Streamlit 1.32+
- **資料來源**：[International football results from 1872 to 2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) + FIFA Rankings

---

## 📦 部署

本專案部署在 [Streamlit Community Cloud](https://streamlit.io/cloud)，每次 push 到 `main` 分支自動重新部署。

線上 Demo：<https://worldcup-ml-2026-bzplzw7hoy7g5dcizaakvt.streamlit.app/>

---

## 📝 授權與作者

- **Author**: AstorYeh (Jay Yeh)
- **Project**: 2026 World Cup ML Prediction · 數據科學期末專題
- **Year**: 2026

---

## 🔗 相關連結

- 🌐 線上 Demo：<https://worldcup-ml-2026-bzplzw7hoy7g5dcizaakvt.streamlit.app/>
- 📦 GitHub：<https://github.com/AstorYeh/worldcup-ml-2026>
- 📑 期末報告 PPT：`presentation/build_ppt.js`（執行 `node build_ppt.js` 即可重新生成）

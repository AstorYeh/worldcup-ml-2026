# 🏆 2026 World Cup ML Prediction System

> 數據科學期末專題 — 用機器學習預測 2026 FIFA 世界盃比賽結果與冠軍

[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org)

## 📊 概覽

| 項目 | 內容 |
|---|---|
| 資料 | **49,328** 場 1872–2026 國際比賽 + **67,894** 筆 FIFA 排名 |
| 分類 | XGBoost（勝/平/負三向）— Walk-Forward 驗證 |
| 迴歸 | Poisson Regression（主客隊進球數）|
| **分群** | **KMeans + PCA — 48 支球隊風格分群（v2.2 新增）** |
| 模擬 | Monte Carlo 3,000 / 10,000 次奪冠路徑模擬 |

## 🚀 快速啟動

```bash
pip install -r requirements.txt
python pretrain.py            # ← 一次性訓練，產生 models/ 內所有 pkl（約 1 分鐘）
streamlit run app.py
```

> **v2.2 重點改動**：把訓練從即時改為**離線預訓練**，模型 / Monte Carlo / 分群結果都存在 `models/*.pkl`。Streamlit 啟動時直接 `load_pretrained()` 載入，**徹底解決 Streamlit Cloud 冷啟動 30 秒等待**問題。

## 📁 專案結構

```
worldcup-ml-2026/
├── app.py                  # Streamlit 主程式（6 個分頁）
├── pretrain.py             # 離線訓練腳本（v2.2 新增）
├── train.py                # 舊版訓練腳本（保留）
├── models/                 # 預訓練模型（v2.2 新增）
│   ├── clf.pkl             # XGBoost 分類器
│   ├── poisson1.pkl        # 主隊進球 Poisson
│   ├── poisson2.pkl        # 客隊進球 Poisson
│   ├── feat_cols.pkl       # 特徵欄位
│   ├── val_accs.pkl        # Walk-Forward 驗證準確率
│   ├── mc_results.pkl      # Monte Carlo 模擬結果
│   └── team_clusters.pkl   # 球隊風格分群
├── figures/                # 15 張視覺化圖
├── notebook/               # 分析 notebook
└── requirements.txt
```

## 🎯 三大預測任務（對應課程要求）

1. **分類** — XGBoost 三向勝負預測，Walk-Forward 驗證跨 4 屆世界盃
2. **迴歸** — Poisson Regression 預測進球數（主客分開模型）
3. **分群** — KMeans + PCA 對 48 隊做攻守風格非監督分群（k=3, silhouette=0.31）

## 📱 介面分頁

| 分頁 | 內容 |
|---|---|
| 📊 專題總覽 | 資料規模、12 個小組分組 |
| 🔮 2026 預測 | 各組逐場比分預測，含 Walk-Forward 驗證表 |
| 📈 數據分析 | 進球分佈、年代趨勢、隊伍實力對照 |
| 🎯 球隊風格分群 | PCA 散點 + 兩隊雷達圖比較（v2.2 新增）|
| 🏅 奪冠預測 | Monte Carlo 奪冠機率 Top 20 |
| 📅 完整賽程 | 含淘汰賽路徑 HTML 整合 |

## 🔧 重要說明

- **第一次跑請先執行 `python pretrain.py`** 產生 `models/` 內所有 pkl，否則新增的分群頁與 Monte Carlo 頁會顯示「找不到 pkl」警告。
- 若 `models/` 內檔案完整，`app.py` 會跳過所有重訓邏輯直接使用快取結果，反應時間 < 1 秒。
- 老師現場驗收：`streamlit run app.py` 即可，所有 5 頁都會用預訓練模型。

## 📝 上傳清單（6/9 課堂）

- [x] 投影片（.pptx + .pdf）
- [x] 程式碼（整個 repo + zip）
- [x] 資料集（`pretrain.py` 內含資料下載邏輯，或附 `data_cache/*.pkl`）
- [x] 預訓練模型（`models/*.pkl`）

---
**Author**: AstorYeh ｜ 數據科學期末專題 ｜ 2026.06.09 簡報

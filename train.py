"""
🏆 模型訓練腳本
World Cup 2026 — Model Training
================================
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 載入資料
# ============================================================
print("=" * 60)
print("1. 載入歷史世界盃資料...")
print("=" * 60)

# 2010-2022 世界盃小組賽 + 淘汰賽（共40場）
data = {
    'year': [2010]*12 + [2014]*12 + [2018]*12 + [2022]*12,
    'team1': [
        # 2010
        'South Africa','Mexico','Uruguay','France','Argentina','Nigeria',
        'South Korea','Greece','Germany','Australia','Spain','Netherlands',
        # 2014
        'Brazil','Croatia','Mexico','Cameroon','Germany','Portugal',
        'France','Honduras','Argentina','Bosnia','Iran','Nigeria',
        # 2018
        'Russia','Saudi Arabia','Egypt','Uruguay','Portugal','Spain',
        'France','Argentina','Brazil','Switzerland','Colombia','Japan',
        # 2022
        'Qatar','Ecuador','Senegal','Netherlands','England','Iran',
        'USA','Wales','Argentina','Mexico','Denmark','Tunisia',
    ],
    'team2': [
        # 2010
        'Mexico','Uruguay','France','South Africa','Korea Republic','Argentina',
        'Greece','Nigeria','Australia','Germany','Netherlands','Spain',
        # 2014
        'Croatia','Mexico','Cameroon','Brazil','Portugal','Germany',
        'Honduras','France','Bosnia','Argentina','Nigeria','Iran',
        # 2018
        'Saudi Arabia','Egypt','Russia','Saudi Arabia','Spain','Portugal',
        'Argentina','France','Switzerland','Brazil','Japan','Colombia',
        # 2022
        'Ecuador','Qatar','Netherlands','Senegal','Iran','England',
        'Wales','USA','Mexico','Argentina','Tunisia','Denmark',
    ],
    'score1': [
        # 2010
        1,0,0,0,1,1,2,2,4,0,0,1,
        # 2014
        3,1,1,0,4,2,3,0,2,1,0,2,
        # 2018
        5,0,1,3,3,3,4,3,1,2,2,2,
        # 2022
        0,2,3,1,6,2,1,1,1,2,0,1,
    ],
    'score2': [
        # 2010
        1,1,0,0,0,0,0,0,0,1,1,1,
        # 2014
        1,1,4,4,0,1,0,0,1,0,1,0,
        # 2018
        1,0,1,1,3,3,3,4,1,1,2,2,
        # 2022
        2,0,3,2,0,6,1,1,2,0,1,0,
    ]
}

df = pd.DataFrame(data)
print(f"總比賽場次: {len(df)}")
print(df.head())

# ============================================================
# 2. 建立特徵
# ============================================================
print("\n" + "=" * 60)
print("2. 特徵工程...")
print("=" * 60)

# FIFA 排名（2025年底）
fifa_ranking = {
    'Spain': 1, 'Argentina': 2, 'France': 3, 'England': 4, 'Brazil': 5,
    'Portugal': 6, 'Netherlands': 7, 'Germany': 8, 'Italy': 9, 'Croatia': 10,
    'Colombia': 11, 'Uruguay': 12, 'Belgium': 13, 'Mexico': 14, 'USA': 15,
    'Japan': 16, 'Morocco': 17, 'Switzerland': 18, 'Korea Republic': 19, 'Australia': 20,
    'Iran': 21, 'Senegal': 22, 'Ecuador': 23, 'Norway': 24, 'Nigeria': 25,
    'Costa Rica': 26, 'Paraguay': 27, 'Austria': 28, 'Cameroon': 29, 'Sweden': 30,
    'Czech Republic': 31, 'Turkey': 32, 'Scotland': 33, 'Ivory Coast': 34, 'Egypt': 35,
    'Algeria': 36, 'Ghana': 37, 'South Africa': 38, 'Saudi Arabia': 39, 'Qatar': 40,
    'Tunisia': 41, 'Jordan': 42, 'Uzbekistan': 43, 'Panama': 44, 'Iraq': 45,
    'New Zealand': 46, 'Cape Verde': 47, 'Haiti': 48,
    'Bosnia': 50, 'Honduras': 55, 'Wales': 60, 'Russia': 65, 'Curacao': 70
}

def create_features(df, fifa_rank):
    features = pd.DataFrame()
    features['rank1'] = df['team1'].map(lambda x: fifa_rank.get(x, 50))
    features['rank2'] = df['team2'].map(lambda x: fifa_rank.get(x, 50))
    features['rank_diff'] = features['rank1'] - features['rank2']
    features['rank_ratio'] = features['rank1'] / features['rank2']
    features['rank_sum'] = features['rank1'] + features['rank2']
    return features

X = create_features(df, fifa_ranking)
y_result = (df['score1'] > df['score2']).astype(int)  # team1 勝負
y_goal1 = df['score1']  # team1 進球數
y_goal2 = df['score2']  # team2 進球數

print("特徵欄位:", list(X.columns))
print("樣本數:", len(X))

# ============================================================
# 3. 訓練模型
# ============================================================
print("\n" + "=" * 60)
print("3. 模型訓練...")
print("=" * 60)

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y_result, test_size=0.2, random_state=42)

# 模型 1: XGBoost
print("\n--- XGBoost ---")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
print(f"XGBoost 準確率: {xgb_acc:.2%}")

# 模型 2: Logistic Regression
print("\n--- Logistic Regression ---")
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
print(f"Logistic Regression 準確率: {lr_acc:.2%}")

# 模型 3: Random Forest
print("\n--- Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"Random Forest 準確率: {rf_acc:.2%}")

# 進球數模型: Poisson Regression
print("\n--- Poisson Regression (Goals) ---")
X_train_g, X_test_g, y1_train, y1_test = train_test_split(X, y_goal1, test_size=0.2, random_state=42)
_, _, y2_train, y2_test = train_test_split(X, y_goal2, test_size=0.2, random_state=42)

poisson1 = PoissonRegressor(alpha=0.1, max_iter=500)
poisson2 = PoissonRegressor(alpha=0.1, max_iter=500)
poisson1.fit(X_train_g, y1_train)
poisson2.fit(X_train_g, y2_train)

p1_mse = mean_squared_error(y1_test, poisson1.predict(X_test_g))
p2_mse = mean_squared_error(y2_test, poisson2.predict(X_test_g))
print(f"Team1 進球 MSE: {p1_mse:.2f}")
print(f"Team2 進球 MSE: {p2_mse:.2f}")

# ============================================================
# 4. Feature Importance
# ============================================================
print("\n" + "=" * 60)
print("4. Feature Importance (XGBoost)")
print("=" * 60)
importance = xgb_model.feature_importances_
feat_names = ['rank1', 'rank2', 'rank_diff', 'rank_ratio', 'rank_sum']
for name, imp in sorted(zip(feat_names, importance), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {imp:.4f}")

# ============================================================
# 5. 儲存模型
# ============================================================
print("\n" + "=" * 60)
print("5. 儲存模型...")
print("=" * 60)

models = {
    'xgb': xgb_model,
    'lr': lr_model,
    'rf': rf_model,
    'poisson1': poisson1,
    'poisson2': poisson2,
    'fifa_rank': fifa_ranking
}

with open('models.pkl', 'wb') as f:
    pickle.dump(models, f)
print("模型已儲存至 models.pkl")

# ============================================================
# 6. 測試預測
# ============================================================
print("\n" + "=" * 60)
print("6. 測試預測：西班牙 vs 法國")
print("=" * 60)

r1, r2 = fifa_ranking['Spain'], fifa_ranking['France']
X_test_match = pd.DataFrame([[r1, r2, r1-r2, r1/r2, r1+r2]],
                              columns=['rank1', 'rank2', 'rank_diff', 'rank_ratio', 'rank_sum'])

win_prob = xgb_model.predict_proba(X_test_match)[0]
print(f"西班牙 FIFA 排名: {r1}")
print(f"法國 FIFA 排名: {r2}")
print(f"西班牙 勝率: {win_prob[1]:.1%}")
print(f"法國 勝率: {win_prob[0]:.1%}")

goal1 = max(0, int(round(poisson1.predict(X_test_match)[0])))
goal2 = max(0, int(round(poisson2.predict(X_test_match)[0])))
print(f"預測比分: {goal1} - {goal2}")

print("\n" + "=" * 60)
print("✅ 訓練完成！")
print("=" * 60)

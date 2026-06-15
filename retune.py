# -*- coding: utf-8 -*-
"""每日微調：抓最新 2026 世界盃賽果 → 重跑 Monte Carlo → 只更新 models/mc_results.pkl。

不重訓 clf/Poisson（沿用既有 pkl），故只有 mc_results.pkl 會因新賽果而變動，
配合 retune.bat 的「有變動才 commit/push」邏輯，避免無意義的每日提交。
執行：python retune.py
"""
import os
import pickle

import pretrain as P


def _load(name):
    with open(os.path.join(P.MODEL_DIR, name), 'rb') as f:
        return pickle.load(f)


def main() -> None:
    match_df, fifa_df = P.load_data()          # load_data 內已自動注入最新 2026 賽果
    clf = _load('clf.pkl')
    p1 = _load('poisson1.pkl')
    p2 = _load('poisson2.pkl')
    fc = _load('feat_cols.pkl')
    P.run_monte_carlo(match_df, fifa_df, clf, p1, p2, fc, n_sims=10000)
    print("[retune] mc_results.pkl 已更新")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
官方锚地版本分析：33 CFR 110.214 Commercial Anchorage F + G
区域来源: https://www.ecfr.gov/current/title-33/chapter-I/subchapter-J/part-110/subpart-B/section-110.214
目标: 预测次日锚地内 AIS 不同船只数 (F+G 合并)
对比: Model A 纯 AIS 滞后特征 vs Model B 加 SAR 过境滞后特征（无泄漏）
切分: 训练 2024-01 .. 2025-06, 测试 2025-07 .. 2025-12
"""
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WORK = Path(__file__).parent
FIG = WORK / "figures"
FIG.mkdir(exist_ok=True)

FULL_START, FULL_END = "2024-01-01", "2025-12-31"
TRAIN_END = "2025-07-01"

# 锚地多边形（lat, lon），与 pull_gfw_anchorages.py 保持一致
def dms(d, m, s):
    return d + m / 60 + s / 3600

ANCHORAGE_F = [
    (dms(33, 43, 5.1), -dms(118, 8, 4.0)), (dms(33, 43, 5.0), -dms(118, 10, 32.5)),
    (dms(33, 42, 13.3), -dms(118, 9, 54.8)), (dms(33, 40, 51.3), -dms(118, 9, 32.2)),
    (dms(33, 38, 36.2), -dms(118, 7, 43.8)), (dms(33, 40, 44.4), -dms(118, 6, 51.4)),
]
ANCHORAGE_G = [
    (dms(33, 43, 5.4), -dms(118, 11, 17.9)), (dms(33, 43, 5.4), -dms(118, 12, 18.6)),
    (dms(33, 42, 25.8), -dms(118, 14, 19.2)), (dms(33, 40, 50.4), -dms(118, 13, 1.2)),
    (dms(33, 41, 2.9), -dms(118, 12, 19.0)), (dms(33, 42, 10.8), -dms(118, 11, 36.0)),
]

# ----------------------------------------------------------------------
ais = pd.read_csv(WORK / "data" / "ais_anchorage_raw.csv", parse_dates=["date"])
sar = pd.read_csv(WORK / "data" / "sar_anchorage_raw.csv", parse_dates=["date"])

# 描述统计
for name, g in ais.groupby("anchorage"):
    print(f"锚地 {name}: {g['date'].nunique()} 天有 AIS 记录, "
          f"日均船只 {g.groupby('date')['vessel_id'].nunique().mean():.1f}")
print(f" SAR 记录: {len(sar)} 行, 过境日 {sar['date'].nunique()} 天, "
      f"其中 F {sar.loc[sar.anchorage=='F','date'].nunique()} 天 / G {sar.loc[sar.anchorage=='G','date'].nunique()} 天")

# 图 0: 锚地位置图（eCFR 官方法规坐标；GFW presence report 不返回 lat/lon，无法叠加实测船位）
fig, ax = plt.subplots(figsize=(7, 6))
for coords, name, color in [(ANCHORAGE_F, "Anchorage F", "#D64541"), (ANCHORAGE_G, "Anchorage G", "#E8A33D")]:
    xs = [lo for la, lo in coords] + [coords[0][1]]
    ys = [la for la, lo in coords] + [coords[0][0]]
    ax.fill(xs, ys, alpha=0.25, color=color, label=f"{name} (33 CFR 110.214)")
    ax.plot(xs, ys, color=color, lw=1.5)
ax.annotate("Long Beach Harbor →", xy=(-118.10, 33.72), fontsize=9, ha="left")
ax.set(xlabel="Longitude", ylabel="Latitude",
       title="Official anchorage grounds F & G (33 CFR 110.214, NAD 83)")
ax.legend(fontsize=9, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "fig0_anchorages_map.png", dpi=140)

# ----------------------------------------------------------------------
# 特征构建（F+G 合并）
ais["vessel_id"] = ais["vessel_id"].fillna("")
ais["gear_type"] = ais["gear_type"].fillna("")


def nuniq(df):
    return df["vessel_id"].replace("", np.nan).nunique()


def daily_features(g):
    return pd.Series({
        "n_vessels": nuniq(g),
        "vessel_hours": g["hours"].sum(),
        "avg_speed": g["speed"].mean(),
        "n_cargo": nuniq(g[g["gear_type"] == "CARGO"]),
        "n_tanker": nuniq(g[g["gear_type"] == "TANKER"]),
    })


# 合并两锚地：同一天同一船只算一次
daily_ais = ais.groupby("date").apply(daily_features, include_groups=False).reset_index()

sar["vessel_id"] = sar["vessel_id"].fillna("")
sar["matched"] = sar["vessel_id"] != ""
daily_sar = sar.groupby("date").apply(
    lambda g: pd.Series({
        "sar_detections": g["detections"].sum(),
        "sar_matched_det": g.loc[g["matched"], "detections"].sum(),
        "sar_unmatched_det": g.loc[~g["matched"], "detections"].sum(),
        "sar_matched_vessels": g.loc[g["matched"], "vessel_id"].nunique(),
    }),
    include_groups=False,
).reset_index()

cal = pd.DataFrame({"date": pd.date_range(FULL_START, FULL_END)})
panel = cal.merge(daily_ais, on="date", how="left").merge(daily_sar, on="date", how="left")
missing_ais = panel["n_vessels"].isna().sum()
print(f"\n锚地合并后 AIS 无船日期数: {missing_ais} / {len(panel)}（这些天空锚地，船只数计 0）")
panel["n_vessels"] = panel["n_vessels"].fillna(0)
panel["vessel_hours"] = panel["vessel_hours"].fillna(0)
panel["n_cargo"] = panel["n_cargo"].fillna(0)
panel["n_tanker"] = panel["n_tanker"].fillna(0)
panel["sar_pass_today"] = panel["sar_detections"].notna().astype(int)
for c in ["sar_detections", "sar_matched_det", "sar_unmatched_det", "sar_matched_vessels"]:
    panel[c] = panel[c].fillna(0)

p = panel.sort_values("date").reset_index(drop=True)
p["dow"] = p["date"].dt.dayofweek
p["month"] = p["date"].dt.month

y = p["n_vessels"].shift(-1)
feat = pd.DataFrame(index=p.index)
feat["n_lag1"] = p["n_vessels"].shift(1)
feat["n_lag7"] = p["n_vessels"].shift(7)
feat["n_mean7"] = p["n_vessels"].rolling(7, min_periods=5).mean().shift(1)
feat["n_mean14"] = p["n_vessels"].rolling(14, min_periods=10).mean().shift(1)
feat["n_mean28"] = p["n_vessels"].rolling(28, min_periods=20).mean().shift(1)
feat["n_std7"] = p["n_vessels"].rolling(7, min_periods=5).std().shift(1)
feat["hours_lag1"] = p["vessel_hours"].shift(1)
feat["cargo_share_lag1"] = (p["n_cargo"] / p["n_vessels"].replace(0, np.nan)).shift(1)
feat["tanker_share_lag1"] = (p["n_tanker"] / p["n_vessels"].replace(0, np.nan)).shift(1)
feat["dow"] = p["dow"].astype(int)
feat["month"] = p["month"].astype(int)
AIS_FEATURES = feat.columns.tolist()

pass_days = p.loc[p["sar_pass_today"] == 1, ["date", "sar_detections", "sar_matched_det",
                                             "sar_unmatched_det", "sar_matched_vessels"]]
last_det, last_match, last_unmatch, last_nv, last_gap = [], [], [], [], []
det_28d = []
j = 0
pass_dates = pass_days["date"].tolist()
for d in p["date"]:
    while j < len(pass_dates) and pass_dates[j] <= d:
        j += 1
    if j > 0:
        last = pass_days.iloc[j - 1]
        last_det.append(last["sar_detections"]); last_match.append(last["sar_matched_det"])
        last_unmatch.append(last["sar_unmatched_det"]); last_nv.append(last["sar_matched_vessels"])
        last_gap.append((d - last["date"]).days)
    else:
        last_det.append(np.nan); last_match.append(np.nan)
        last_unmatch.append(np.nan); last_nv.append(np.nan); last_gap.append(np.nan)
    det_28d.append(pass_days.loc[(pass_days["date"] > d - pd.Timedelta(days=28)) &
                                 (pass_days["date"] <= d), "sar_detections"].sum())

feat["sar_pass_today"] = p["sar_pass_today"]
feat["days_since_pass"] = last_gap
feat["last_pass_det"] = last_det
feat["last_pass_matched"] = last_match
feat["last_pass_unmatched"] = last_unmatch
feat["last_pass_vessels"] = last_nv
feat["det_28d"] = det_28d
SAR_FEATURES = ["sar_pass_today", "days_since_pass", "last_pass_det", "last_pass_matched",
                "last_pass_unmatched", "last_pass_vessels", "det_28d"]

def build_data(horizon):
    d = feat.copy()
    d["y"] = p["n_vessels"].shift(-horizon)
    d["date"] = p["date"]
    d = d.dropna(subset=["y", "n_lag7", "n_mean28"]).reset_index(drop=True)
    d["days_since_pass"] = d["days_since_pass"].fillna(60).clip(0, 60)
    for c in ["last_pass_det", "last_pass_matched", "last_pass_unmatched", "last_pass_vessels"]:
        d[c] = d[c].fillna(0)
    d["cargo_share_lag1"] = d["cargo_share_lag1"].fillna(0)
    d["tanker_share_lag1"] = d["tanker_share_lag1"].fillna(0)
    return d


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


models = {
    "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "HGB": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                 max_depth=3, random_state=0),
}

HORIZONS = [1, 7]
all_res = []
for h in HORIZONS:
    data = build_data(h)
    if h == 1:
        data.to_csv(WORK / "data" / "panel_anchorage.csv", index=False)

    train = data[data["date"] < TRAIN_END]
    test = data[data["date"] >= TRAIN_END]
    y_tr, y_te = train["y"], test["y"]

    results = []
    add = lambda name, pred: results.append(
        {"horizon": h, "model": name, "MAE": round(mae(y_te, pred), 3),
         "RMSE": round(rmse(y_te, pred), 3)})

    add(f"Baseline 持续性(h天后=今天)", test["n_lag1"])
    for mname, mk in models.items():
        mA = mk(); mA.fit(train[AIS_FEATURES], y_tr)
        add(f"Model A ({mname}, 纯AIS)", mA.predict(test[AIS_FEATURES]))
        mB = mk(); mB.fit(train[AIS_FEATURES + SAR_FEATURES], y_tr)
        add(f"Model B ({mname}, AIS+SAR)", mB.predict(test[AIS_FEATURES + SAR_FEATURES]))

    res = pd.DataFrame(results)
    all_res.append(res)
    print(f"\n===== 测试集表现 (2025-07 至 2025-12, 预测 {h} 天后锚地船只数 F+G, "
          f"样本 {len(data)}) =====")
    print(res[["model", "MAE", "RMSE"]].to_string(index=False))
    res.to_csv(WORK / "data" / f"model_comparison_anchorage_h{h}.csv", index=False)

# 图 1: 锚地时间序列 + SAR 过境
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(p["date"], p["n_vessels"], lw=1, color="#167D8D", label="AIS daily vessels (F+G)")
sp = p[p["sar_pass_today"] == 1]
ax.scatter(sp["date"], sp["n_vessels"], color="#D64541", zorder=5, s=30,
           label=f"SAR pass day (n={len(sp)})")
ax.set(title="Official anchorages F+G: anchored vessel count and SAR pass days, 2024-2025",
       ylabel="anchored AIS vessels")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig1_anchorage_timeseries.png", dpi=140)

# 图 2: 每次过境 SAR 检测构成
sp2 = sp.copy()
x = np.arange(len(sp2))
fig, ax = plt.subplots(figsize=(13, 4.2))
ax.bar(x, sp2["sar_matched_det"], width=0.8, color="#167D8D", label="matched to AIS")
ax.bar(x, sp2["sar_unmatched_det"], width=0.8, bottom=sp2["sar_matched_det"],
       color="#D64541", label="unmatched")
ax.set_xticks(x[::8])
ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in sp2["date"][::8]], rotation=45, ha="right", fontsize=8)
ax.set(title="SAR detections per pass in anchorages F+G, by AIS match status")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig2_anchorage_sar.png", dpi=140)

print("\nSAR 检测构成: 匹配", int(p["sar_matched_det"].sum()),
      "未匹配", int(p["sar_unmatched_det"].sum()),
      "| 过境频率: 每", round(len(p) / max(len(sp), 1), 1), "天一次")
print("图已保存到 figures/")

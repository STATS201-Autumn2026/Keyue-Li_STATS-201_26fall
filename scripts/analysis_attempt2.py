# -*- coding: utf-8 -*-
"""Attempt 2 additions: per-site robustness (F vs G) and 2024 Q4 walk-forward evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = Path(__file__).parent
FIG = WORK / "figures"
FIG.mkdir(exist_ok=True)
OUT = WORK / "data"

AIS_F, AIS_G = "n_lag1", "n_lag7"  # noqa: F841 (placeholder to keep ruff quiet)
AIS_COLS = ["n_lag1", "n_lag7", "n_mean7", "n_mean14", "n_mean28", "n_std7",
            "hours_lag1", "cargo_share_lag1", "tanker_share_lag1", "dow", "month"]
SAR_COLS = ["sar_pass_today", "days_since_pass", "last_pass_det", "last_pass_matched",
            "last_pass_unmatched", "last_pass_vessels", "det_28d"]
FULL_START, FULL_END = "2024-01-01", "2025-12-31"


def load_site(site):
    ais = pd.read_csv(WORK / "data/ais_anchorage_raw.csv", parse_dates=["date"])
    sar = pd.read_csv(WORK / "data/sar_anchorage_raw.csv", parse_dates=["date"])
    ais = ais[ais.anchorage == site]
    sar = sar[sar.anchorage == site]

    idx = pd.date_range(FULL_START, FULL_END, freq="D")
    daily = (ais.groupby("date").agg(vessels=("vessel_id", "nunique"),
                                     hours=("hours", "sum"))
               .reindex(idx).fillna(0.0))
    daily.index.name = "date"

    # per-pass SAR features
    sar["matched"] = sar.vessel_id.notna().astype(int)
    passf = (sar.groupby("date").agg(det=("date", "size"),
                                     matched=("matched", "sum"),
                                     vessels=("vessel_id", "nunique")))
    passf["unmatched"] = passf.det - passf.matched
    daily["sar_pass_today"] = daily.index.isin(passf.index).astype(int)

    det_map, mat_map, unm_map, ves_map = {}, {}, {}, {}
    days_since, last_det, last_mat, last_unm, last_ves = [], [], [], [], []
    prev_pass_dates = []
    for d in daily.index:
        past = passf[passf.index <= d]
        if len(past):
            lp = past.index[-1]
            days_since.append((d - lp).days)
            last_det.append(past.det.loc[lp]); last_mat.append(past.matched.loc[lp])
            last_unm.append(past.unmatched.loc[lp]); last_ves.append(past.vessels.loc[lp])
            det_28d = past.det[past.index > d - pd.Timedelta(days=28)].sum()
        else:
            days_since.append(np.nan); last_det.append(np.nan); last_mat.append(np.nan)
            last_unm.append(np.nan); last_ves.append(np.nan); det_28d = np.nan
        daily.loc[d, "det_28d"] = det_28d
    daily["days_since_pass"] = days_since
    daily["last_pass_det"] = last_det
    daily["last_pass_matched"] = last_mat
    daily["last_pass_unmatched"] = last_unm
    daily["last_pass_vessels"] = last_ves

    n = daily["vessels"]
    daily["n_lag1"] = n.shift(1); daily["n_lag7"] = n.shift(7)
    daily["n_mean7"] = n.rolling(7).mean().shift(1)
    daily["n_mean14"] = n.rolling(14).mean().shift(1)
    daily["n_mean28"] = n.rolling(28).mean().shift(1)
    daily["n_std7"] = n.rolling(7).std().shift(1)
    vtype = ais.pivot_table(index="date", columns="vessel_type", values="hours",
                            aggfunc="sum").reindex(idx).fillna(0.0)
    tot = vtype.sum(axis=1).replace(0, np.nan)
    daily["hours_lag1"] = daily["hours"].shift(1)
    daily["cargo_share_lag1"] = (vtype.get("CARGO", 0) / tot).shift(1)
    daily["tanker_share_lag1"] = (vtype.get("TANKER", 0) / tot).shift(1)
    daily["dow"] = daily.index.dayofweek
    daily["month"] = daily.index.month
    daily["y"] = daily["vessels"]
    panel = daily.dropna(subset=["n_mean28", "hours_lag1", "days_since_pass",
                                 "cargo_share_lag1", "tanker_share_lag1",
                                 "det_28d", "last_pass_det"]).reset_index()
    return panel


def fit_eval(panel, train_end, test_start, test_end):
    tr = panel[panel.date <= train_end]
    te = panel[(panel.date >= test_start) & (panel.date <= test_end)]
    rows = []
    for name, cols in [("A", AIS_COLS), ("B", AIS_COLS + SAR_COLS)]:
        Xa = make_pipeline(StandardScaler(), Ridge())
        Xa.fit(tr[cols], tr.y)
        pred = Xa.predict(te[cols])
        rows.append({"model": name, "MAE": np.abs(te.y - pred).mean(),
                     "RMSE": np.sqrt(((te.y - pred) ** 2).mean()),
                     "n_test": len(te)})
        base = te.y.shift(0)  # persistence handled per horizon by caller; h=1 baseline = y_lag1
    return pd.DataFrame(rows), te


def main():
    results = {}
    for site in ["F", "G"]:
        p = load_site(site)
        p.to_csv(OUT / f"panel_anchorage_{site}.csv", index=False)
        r, _ = fit_eval(p, "2025-06-30", "2025-07-01", FULL_END)
        results[site] = r
        print(f"--- site {site} (n_panel={len(p)}, avg daily vessels={p.y.mean():.1f}) ---")
        print(r.to_string(index=False))

    # combined panel Q4 walk-forward: reuse released combined panel for consistency
    comb = pd.read_csv(WORK / "data/panel_anchorage.csv", parse_dates=["date"])
    r_q4, te = fit_eval(comb, "2024-08-15", "2024-08-16", "2024-12-31")
    r_q4["horizon"] = 1
    # persistence baseline h=1 on same window
    base_mae = (te.y - te.n_lag1).abs().mean()
    r_q4["split"] = "Q4 walk-forward"
    print(f"--- Q4 walk-forward (train<=2024-08-15, test 2024-08-16..12-31), n_test={len(te)} ---")
    print(r_q4.to_string(index=False))
    print(f"persistence baseline MAE (h=1) on Q4 window: {base_mae:.3f}")
    print(f"Q4 window mean y = {te.y.mean():.2f}, zero days = {(te.y == 0).mean():.1%}")

    # combined main split for reference
    r_main, _ = fit_eval(comb, "2025-06-30", "2025-07-01", FULL_END)
    r_main["split"] = "main"
    base_main = (comb[comb.date >= "2025-07-01"].y - comb[comb.date >= "2025-07-01"].n_lag1).abs().mean()
    print("--- main split (reference) ---")
    print(r_main.to_string(index=False))
    print(f"persistence baseline MAE (h=1) main: {base_main:.3f}")

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    groups = [("F", results["F"]), ("G", results["G"]), ("Q4 window", r_q4)]
    for ax, (label, r) in zip(axes, groups):
        x = np.arange(len(r))
        ax.bar(x - 0.18, r.MAE, width=0.36, label="Model A (AIS)")
        ax.bar(x + 0.18, r.MAE, width=0.36, color="#d62728", label="Model B (AIS+SAR)")
        for i, v in enumerate(r.MAE):
            ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(r.model)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Ridge, h = 1", fontsize=9)
    axes[0].set_ylabel("Test MAE (vessels)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Attempt 2 robustness checks: per-site models and 2024 Q4 regime-switch window",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_robustness_checks.png", dpi=200)
    print("figure saved:", FIG / "fig5_robustness_checks.png")


if __name__ == "__main__":
    main()

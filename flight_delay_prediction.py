"""
================================================================================
FLUGVERSPÄTUNGS-PROGNOSE – United Airlines @ JFK
Production-Ready Version mit Dashboard (8 Plots)
================================================================================
"""

import os
import json
import yaml
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import lightgbm as lgb
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

ARR_PATH = config["paths"]["input"]["arrivals"]
DEP_PATH = config["paths"]["input"]["departures"]
OUT_PLOT = config["paths"]["output"]["plots"]
OUT_MODEL = config["paths"]["output"]["model"]
OUT_METRICS = config["paths"]["output"]["metrics"]

os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)

sns.set_style("whitegrid")

COLORS = {
    "primary": "#1B4F72",
    "secondary": "#2E86C1",
    "accent": "#E74C3C",
    "ok": "#27AE60",
    "warn": "#F39C12"
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Lade Daten...")

arr = pd.read_csv(ARR_PATH, skiprows=7)
dep = pd.read_csv(DEP_PATH, skiprows=7)

arr.columns = arr.columns.str.strip()
dep.columns = dep.columns.str.strip()

arr = arr.dropna(subset=["Date (MM/DD/YYYY)"])
dep = dep.dropna(subset=["Date (MM/DD/YYYY)"])

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def build_features(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["Date (MM/DD/YYYY)"], format="%m/%d/%Y")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["season"] = df["month"].map({
        12:0,1:0,2:0,
        3:1,4:1,5:1,
        6:2,7:2,8:2,
        9:3,10:3,11:3
    })

    df["sched_time"] = pd.to_datetime(
        df["Scheduled departure time"],
        format="%H:%M",
        errors="coerce"
    )

    df["hour_of_day"] = df["sched_time"].dt.hour
    df["is_peak_hour"] = df["hour_of_day"].between(14, 20).astype(int)

    df["route"] = df["Destination Airport"].fillna("UNKNOWN").astype("category")

    df["raw_delay"] = df["Departure delay (Minutes)"]
    df["target"] = (df["raw_delay"] >= 15).astype(int)

    df = df.sort_values("date").reset_index(drop=True)

    df["carrier_hist_delay"] = (
        df.groupby("Carrier Code")["raw_delay"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0)
    )

    return df

dep_feat = build_features(dep)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "year", "month", "day_of_week", "is_weekend",
    "season", "hour_of_day", "is_peak_hour",
    "carrier_hist_delay", "route"
]

# ─────────────────────────────────────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────────────────────────────────────
split_year = config["split"]["train_until_year"]

train = dep_feat[dep_feat["year"] <= split_year]
test  = dep_feat[dep_feat["year"] > split_year]

X_train = train[FEATURE_COLS]
y_train = train["target"]
X_test  = test[FEATURE_COLS]
y_test  = test["target"]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("Trainiere Modell...")

params = config["lightgbm"]["optimized"]
params.update({
    "objective": "binary",
    "metric": "auc",
    "verbose": -1,
    "random_state": 42
})

lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=["route"])
lgb_val   = lgb.Dataset(X_test, label=y_test, categorical_feature=["route"])

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(40, verbose=False)]
)

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
y_pred_proba = model.predict(X_test)

thresholds = np.arange(
    config["model"]["threshold_scan"]["start"],
    config["model"]["threshold_scan"]["end"],
    config["model"]["threshold_scan"]["step"]
)

f1_scores = [f1_score(y_test, (y_pred_proba >= t).astype(int)) for t in thresholds]
best_thresh = thresholds[np.argmax(f1_scores)]

y_pred = (y_pred_proba >= best_thresh).astype(int)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"AUC: {auc:.4f}")
print(f"Best Threshold: {best_thresh:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD (8 PLOTS)
# ─────────────────────────────────────────────────────────────────────────────
print("Erstelle Dashboard...")

fig = plt.figure(figsize=(20, 22))
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

# 1 Histogram
ax1 = fig.add_subplot(gs[0, 0])
delays = dep_feat["raw_delay"].clip(-30, 200)

ax1.hist(delays[delays < 0], bins=40, color=COLORS["ok"], alpha=0.7)
ax1.hist(delays[(delays >= 0) & (delays < 15)], bins=20, color=COLORS["warn"], alpha=0.7)
ax1.hist(delays[delays >= 15], bins=60, color=COLORS["accent"], alpha=0.7)

ax1.axvline(15, linestyle="--", color="black")
ax1.set_title("Verteilung der Abflugverspätungen")

# 2 Monat
ax2 = fig.add_subplot(gs[0, 1])
(dep_feat.groupby("month")["target"].mean() * 100).plot(kind="bar", ax=ax2, color=COLORS["secondary"])
ax2.set_title("Verspätungsrate nach Monat")


# 3 Tageszeit
ax3 = fig.add_subplot(gs[1, 0])

hour = dep_feat.groupby("hour_of_day")["target"].mean() * 100

ax3.plot(hour.index, hour.values, color=COLORS["primary"])
ax3.fill_between(hour.index, hour.values, alpha=0.3)

ax3.set_title("Verspätungsrate nach Tageszeit")


# 4 Feature Importance
ax4 = fig.add_subplot(gs[1, 1])
fi = pd.Series(model.feature_importance(), index=FEATURE_COLS).sort_values()
ax4.barh(fi.index, fi.values, color=COLORS["secondary"])
ax4.set_title("Feature Importance")

# 5 ROC
ax5 = fig.add_subplot(gs[2, 0])
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax5.plot(fpr, tpr, label=f"AUC={auc:.3f}")
ax5.plot([0,1],[0,1],"k--")
ax5.legend()

# 6 Confusion
ax6 = fig.add_subplot(gs[2, 1])
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=",d", ax=ax6)

# 7 PR Curve
ax7 = fig.add_subplot(gs[3, 0])
prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
ax7.plot(rec, prec)

# 8 Wochentag
ax8 = fig.add_subplot(gs[3, 1])
dow = dep_feat.groupby("day_of_week")["target"].mean() * 100
colors = [COLORS["accent"] if i in [4,6] else COLORS["secondary"] for i in range(7)]
ax8.bar(range(7), dow.values, color=colors)

fig.suptitle("Flugverspätungs-Prognose Dashboard", fontsize=16)

fig.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
model.save_model(OUT_MODEL)

with open(OUT_METRICS, "w") as f:
    json.dump({"auc": float(auc), "threshold": float(best_thresh)}, f)

print("\n✅ Alles erfolgreich abgeschlossen!")

"""
================================================================================
FLUGVERSPÄTUNGS-PROGNOSE – United Airlines @ JFK
Production-Ready Version mit:
  - Train / Validation (2014–2015) / Test (2021–2022) Split
  - Modell-Vergleich: LightGBM vs CatBoost vs XGBoost
  - Accuracy + vollständige Metriken
  - 8-Plot Dashboard mit Achsenbeschriftungen & Überschriften
================================================================================
"""

import os, json, yaml, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, accuracy_score
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

ARR_PATH    = cfg["paths"]["input"]["arrivals"]
DEP_PATH    = cfg["paths"]["input"]["departures"]
OUT_PLOT    = cfg["paths"]["output"]["plots"]
OUT_MODEL   = cfg["paths"]["output"]["model"]
OUT_METRICS = cfg["paths"]["output"]["metrics"]
os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)

sns.set_style("whitegrid")
COLORS = {
    "primary":   "#1B4F72",
    "secondary": "#2E86C1",
    "accent":    "#E74C3C",
    "ok":        "#27AE60",
    "warn":      "#F39C12",
    "lgb":       "#2E86C1",
    "cat":       "#8E44AD",
    "xgb":       "#E74C3C",
}
MONTH_LABELS = ["Jan","Feb","Mär","Apr","Mai","Jun",
                "Jul","Aug","Sep","Okt","Nov","Dez"]
DOW_LABELS   = ["Mo","Di","Mi","Do","Fr","Sa","So"]

TRAIN_UNTIL = cfg["split"]["train_until_year"]   # 2013
VAL_YEARS   = cfg["split"]["validation_years"]   # [2014, 2015]
TEST_FROM   = cfg["split"]["test_from_year"]     # 2021
VAL_LABEL   = f"{VAL_YEARS[0]}–{VAL_YEARS[-1]}"

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATEN LADEN  (Schritte 1–3)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SCHRITTE 1–3  |  Daten laden & analysieren")
print("=" * 65)

dep = pd.read_csv(DEP_PATH, skiprows=7)
dep.columns = dep.columns.str.strip()
dep = dep.dropna(subset=["Date (MM/DD/YYYY)"])

delay_col = "Departure delay (Minutes)"
pct_early  = (dep[delay_col] < 0).mean()  * 100
pct_slight = ((dep[delay_col] >= 0) & (dep[delay_col] < 15)).mean() * 100
pct_late   = (dep[delay_col] >= 15).mean() * 100
print(f"Abflüge gesamt  : {len(dep):,} Zeilen")
print(f"Frühzeitig (< 0 min)  : {pct_early:.1f}%")
print(f"Leicht spät (0–14 min): {pct_slight:.1f}%")
print(f"Verspätet  (≥15 min)  : {pct_late:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING  (Schritt 4)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SCHRITT 4  |  Feature Engineering")
print("=" * 65)

def build_features(df):
    df = df.copy()
    df["date"]        = pd.to_datetime(df["Date (MM/DD/YYYY)"], format="%m/%d/%Y")
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["season"]      = df["month"].map({
        12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3
    })
    df["sched_time"]   = pd.to_datetime(df["Scheduled departure time"], format="%H:%M", errors="coerce")
    df["hour_of_day"]  = df["sched_time"].dt.hour
    df["is_peak_hour"] = df["hour_of_day"].between(14, 20).astype(int)

    # Verspätungsursachen (wichtige Prädiktoren)
    df["delay_carrier"]  = df["Delay Carrier (Minutes)"].fillna(0)
    df["delay_weather"]  = df["Delay Weather (Minutes)"].fillna(0)
    df["delay_nas"]      = df["Delay National Aviation System (Minutes)"].fillna(0)
    df["delay_security"] = df["Delay Security (Minutes)"].fillna(0)
    df["delay_lateac"]   = df["Delay Late Aircraft Arrival (Minutes)"].fillna(0)
    df["taxi_time"]      = df["Taxi-Out time (Minutes)"].fillna(df["Taxi-Out time (Minutes)"].median())

    df["route"]      = df["Destination Airport"].fillna("UNKNOWN").astype(str)
    df["raw_delay"]  = df[delay_col]
    df["target"]     = (df["raw_delay"] >= 15).astype(int)

    df = df.sort_values("date").reset_index(drop=True)
    df["carrier_hist_delay"] = (
        df.groupby("Carrier Code")["raw_delay"]
          .transform(lambda x: x.shift(1).expanding().mean())
          .fillna(0)
    )
    return df

dep_feat = build_features(dep)

# Route label-encode für LGB & XGB
le = LabelEncoder()
dep_feat["route_enc"] = le.fit_transform(dep_feat["route"])

FEAT_NUM = [  # LightGBM & XGBoost (numerisch)
    "year","month","day_of_week","is_weekend","season",
    "hour_of_day","is_peak_hour",
    "delay_carrier","delay_weather","delay_nas","delay_security","delay_lateac",
    "taxi_time","carrier_hist_delay","route_enc"
]
FEAT_CAT = [  # CatBoost (kategorisch nativ)
    "year","month","day_of_week","is_weekend","season",
    "hour_of_day","is_peak_hour",
    "delay_carrier","delay_weather","delay_nas","delay_security","delay_lateac",
    "taxi_time","carrier_hist_delay","route"
]

print(f"Features: {len(FEAT_NUM)} Spalten  |  Shape: {dep_feat.shape}")
print(f"Klassen:  Pünktlich {(dep_feat['target']==0).mean()*100:.1f}%  |  Verspätet {(dep_feat['target']==1).mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SPLIT  (Schritt 5)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SCHRITT 5  |  Train / Validation / Test Split")
print("=" * 65)

train = dep_feat[dep_feat["year"] <= TRAIN_UNTIL]
val   = dep_feat[dep_feat["year"].isin(VAL_YEARS)]
test  = dep_feat[dep_feat["year"] >= TEST_FROM]

X_tr_num,  y_tr  = train[FEAT_NUM], train["target"]
X_val_num, y_val = val[FEAT_NUM],   val["target"]
X_te_num,  y_te  = test[FEAT_NUM],  test["target"]

X_tr_cat  = train[FEAT_CAT]
X_val_cat = val[FEAT_CAT]
X_te_cat  = test[FEAT_CAT]

print(f"Train      : 1987–{TRAIN_UNTIL}  → {len(train):>7,} Zeilen")
print(f"Validation : {VAL_LABEL}        → {len(val):>7,} Zeilen  ← Modellvergleich")
print(f"Test       : {TEST_FROM}–2022    → {len(test):>7,} Zeilen  ← finale Evaluation")
print(f"\nZiel: Wird ein Flug ≥15 min verspätet abfliegen?")

# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktion: optimaler Threshold via F1-Scan
# ─────────────────────────────────────────────────────────────────────────────
def optimize_threshold(y_true, y_proba):
    thresholds = np.arange(
        cfg["model"]["threshold_scan"]["start"],
        cfg["model"]["threshold_scan"]["end"],
        cfg["model"]["threshold_scan"]["step"]
    )
    f1_vals = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
               for t in thresholds]
    best_t = thresholds[np.argmax(f1_vals)]
    y_pred = (y_proba >= best_t).astype(int)
    return {
        "threshold": float(best_t),
        "auc":       float(roc_auc_score(y_true, y_proba)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "y_pred":    y_pred,
        "y_proba":   y_proba,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODELL-VERGLEICH AUF VALIDATION  (Schritte 5 & 6)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"SCHRITTE 5 & 6  |  Modell-Vergleich auf Validation-Set ({VAL_LABEL})")
print("=" * 65)

pos_weight = y_tr.value_counts()[0] / y_tr.value_counts()[1]
results_val = {}

# ── LightGBM ──────────────────────────────────────────────────────────────
print("\n[1/3] LightGBM …")
lgb_params = {
    "objective": "binary", "metric": "auc", "verbose": -1,
    "random_state": 42,
    "learning_rate": 0.03, "num_leaves": 127,
    "min_child_samples": 30, "feature_pre_filter": False,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "lambda_l1": 0.1, "lambda_l2": 0.1,
    "scale_pos_weight": pos_weight,
}
ds_tr_lgb  = lgb.Dataset(X_tr_num,  label=y_tr)
ds_val_lgb = lgb.Dataset(X_val_num, label=y_val)
lgb_model = lgb.train(
    lgb_params, ds_tr_lgb, num_boost_round=500,
    valid_sets=[ds_val_lgb],
    callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)]
)
r = optimize_threshold(y_val, lgb_model.predict(X_val_num))
r["model"] = lgb_model
results_val["LightGBM"] = r
print(f"  AUC={r['auc']:.4f}  Accuracy={r['accuracy']*100:.2f}%  F1={r['f1']:.4f}  Threshold={r['threshold']:.2f}")

# ── CatBoost ──────────────────────────────────────────────────────────────
print("\n[2/3] CatBoost …")
cat_model = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=8,
    eval_metric="AUC", early_stopping_rounds=40,
    class_weights=[1, pos_weight],
    cat_features=["route"], random_seed=42, verbose=0
)
cat_model.fit(X_tr_cat, y_tr, eval_set=(X_val_cat, y_val))
r = optimize_threshold(y_val, cat_model.predict_proba(X_val_cat)[:, 1])
r["model"] = cat_model
results_val["CatBoost"] = r
print(f"  AUC={r['auc']:.4f}  Accuracy={r['accuracy']*100:.2f}%  F1={r['f1']:.4f}  Threshold={r['threshold']:.2f}")

# ── XGBoost ───────────────────────────────────────────────────────────────
print("\n[3/3] XGBoost …")
xgb_model = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.03, max_depth=7,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    scale_pos_weight=pos_weight,
    eval_metric="auc", early_stopping_rounds=40,
    random_state=42, verbosity=0
)
xgb_model.fit(X_tr_num, y_tr, eval_set=[(X_val_num, y_val)], verbose=False)
r = optimize_threshold(y_val, xgb_model.predict_proba(X_val_num)[:, 1])
r["model"] = xgb_model
results_val["XGBoost"] = r
print(f"  AUC={r['auc']:.4f}  Accuracy={r['accuracy']*100:.2f}%  F1={r['f1']:.4f}  Threshold={r['threshold']:.2f}")

best_val = max(results_val, key=lambda k: results_val[k]["auc"])
print(f"\n🏆  Bestes Modell (Validation): {best_val}  AUC={results_val[best_val]['auc']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FINALE EVALUATION AUF TEST-SET  (Schritt 7)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"SCHRITT 7  |  Finale Evaluation auf Test-Set ({TEST_FROM}+)")
print("=" * 65)

results_test = {}
for name in results_val:
    m = results_val[name]["model"]
    if name == "LightGBM":
        proba = m.predict(X_te_num)
    elif name == "CatBoost":
        proba = m.predict_proba(X_te_cat)[:, 1]
    else:
        proba = m.predict_proba(X_te_num)[:, 1]

    r = optimize_threshold(y_te, proba)
    results_test[name] = r

    cm = confusion_matrix(y_te, r["y_pred"])
    tn, fp, fn, tp = cm.ravel()
    print(f"\n── {name} ──")
    print(f"  Threshold  : {r['threshold']:.2f}")
    print(f"  AUC-ROC    : {r['auc']*100:.2f}%")
    print(f"  Accuracy   : {r['accuracy']*100:.2f}%")
    print(f"  F1-Score   : {r['f1']*100:.2f}%")
    print(f"  True  Positives (korrekt verspätet erkannt): {tp:,}")
    print(f"  False Positives (fälschlich gewarnt)       : {fp:,}")
    print(f"  False Negatives (verpasste Verspätungen)   : {fn:,}")
    print(classification_report(y_te, r["y_pred"], target_names=["Pünktlich","Verspätet"]))

best_test = max(results_test, key=lambda k: results_test[k]["auc"])

# ─────────────────────────────────────────────────────────────────────────────
# 6. DASHBOARD (8 PLOTS)  (Schritt 8)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SCHRITT 8  |  Dashboard erstellen")
print("=" * 65)

FS_TITLE = 12
FS_AXIS  = 10
FS_TICK  = 9
MODEL_COLORS = {"LightGBM": COLORS["lgb"], "CatBoost": COLORS["cat"], "XGBoost": COLORS["xgb"]}

fig = plt.figure(figsize=(20, 24))
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.35)

# ── Plot 1: Verspätungsverteilung ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
delays = dep_feat["raw_delay"].clip(-30, 200)
ax1.hist(delays[delays < 0],  bins=40, color=COLORS["ok"],      alpha=0.75, label="Frühzeitig (< 0 min)")
ax1.hist(delays[(delays >= 0) & (delays < 15)], bins=20,
         color=COLORS["warn"], alpha=0.75, label="Leicht spät (0–14 min)")
ax1.hist(delays[delays >= 15], bins=60, color=COLORS["accent"], alpha=0.75, label="Verspätet (≥ 15 min)")
ax1.axvline(15, color="black", linestyle="--", linewidth=1.5, label="Schwelle: 15 min")
ax1.set_title("Verteilung der Abflugverspätungen (alle Flüge)", fontsize=FS_TITLE, fontweight="bold", pad=10)
ax1.set_xlabel("Verspätung in Minuten (begrenzt auf −30 bis +200)", fontsize=FS_AXIS)
ax1.set_ylabel("Anzahl Flüge", fontsize=FS_AXIS)
ax1.tick_params(labelsize=FS_TICK)
ax1.legend(fontsize=8)

# ── Plot 2: Verspätungsrate nach Monat ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
month_rate = dep_feat.groupby("month")["target"].mean() * 100
bar_colors = [COLORS["accent"] if v > 15 else COLORS["secondary"] for v in month_rate.values]
bars = ax2.bar(month_rate.index, month_rate.values, color=bar_colors, edgecolor="white")
for bar, val in zip(bars, month_rate.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
ax2.set_title("Verspätungsrate nach Monat (≥ 15 min)", fontsize=FS_TITLE, fontweight="bold", pad=10)
ax2.set_xlabel("Monat", fontsize=FS_AXIS)
ax2.set_ylabel("Anteil verspäteter Flüge (%)", fontsize=FS_AXIS)
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(MONTH_LABELS, fontsize=FS_TICK, rotation=30)
ax2.tick_params(axis="y", labelsize=FS_TICK)

# ── Plot 3: Verspätungsrate nach Tageszeit ────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
hour_rate = dep_feat.groupby("hour_of_day")["target"].mean() * 100
ax3.fill_between(hour_rate.index, hour_rate.values, alpha=0.2, color=COLORS["secondary"])
ax3.plot(hour_rate.index, hour_rate.values, color=COLORS["primary"], linewidth=2, marker="o", markersize=4)
ax3.axvspan(14, 20, alpha=0.10, color=COLORS["accent"], label="Peak-Zone (14–20 Uhr)")
ax3.set_title("Verspätungsrate nach Tageszeit", fontsize=FS_TITLE, fontweight="bold", pad=10)
ax3.set_xlabel("Geplante Abflugstunde (0 = Mitternacht, 23 = 23 Uhr)", fontsize=FS_AXIS)
ax3.set_ylabel("Anteil verspäteter Flüge (%)", fontsize=FS_AXIS)
ax3.set_xticks(range(0, 24, 2))
ax3.tick_params(labelsize=FS_TICK)
ax3.legend(fontsize=8)

# ── Plot 4: Modellvergleich Validation-Set ────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
names = list(results_val.keys())
metric_groups = {
    "AUC-ROC":  [results_val[n]["auc"]      * 100 for n in names],
    "Accuracy": [results_val[n]["accuracy"] * 100 for n in names],
    "F1-Score": [results_val[n]["f1"]       * 100 for n in names],
}
x4 = np.arange(len(names))
w4 = 0.25
bar_colors4 = [COLORS["primary"], COLORS["ok"], COLORS["warn"]]
for i, (metric, vals) in enumerate(metric_groups.items()):
    b = ax4.bar(x4 + i*w4, vals, w4, label=metric, color=bar_colors4[i], alpha=0.85, edgecolor="white")
    for rect, v in zip(b, vals):
        ax4.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.3,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=7)
ax4.set_title(f"Modellvergleich – Validation-Set ({VAL_LABEL})", fontsize=FS_TITLE, fontweight="bold", pad=10)
ax4.set_xlabel("Modell-Algorithmus", fontsize=FS_AXIS)
ax4.set_ylabel("Score (%)", fontsize=FS_AXIS)
ax4.set_xticks(x4 + w4)
ax4.set_xticklabels(names, fontsize=FS_TICK)
ax4.set_ylim(0, 115)
ax4.tick_params(axis="y", labelsize=FS_TICK)
ax4.legend(fontsize=8)

# ── Plot 5: ROC-Kurven (Test-Set) ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
for name, res in results_test.items():
    fpr, tpr, _ = roc_curve(y_te, res["y_proba"])
    ax5.plot(fpr, tpr, linewidth=2, color=MODEL_COLORS[name],
             label=f"{name}  (AUC = {res['auc']:.3f})")
ax5.plot([0,1],[0,1], "k--", linewidth=1, label="Zufälliges Modell (AUC = 0.5)")
ax5.set_title(f"ROC-Kurven – Alle Modelle (Test-Set {TEST_FROM}–2022)", fontsize=FS_TITLE, fontweight="bold", pad=10)
ax5.set_xlabel("False Positive Rate  (FPR = Fehlalarme / alle Pünktlichen)", fontsize=FS_AXIS)
ax5.set_ylabel("True Positive Rate  (TPR = Erkannte Verspätungen / alle Verspäteten)", fontsize=FS_AXIS)
ax5.tick_params(labelsize=FS_TICK)
ax5.legend(fontsize=8)

# ── Plot 6: Konfusionsmatrix (bestes Modell) ──────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
cm = confusion_matrix(y_te, results_test[best_test]["y_pred"])
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax6,
            xticklabels=["Pünktlich","Verspätet"],
            yticklabels=["Pünktlich","Verspätet"],
            annot_kws={"size": 12})
tn, fp, fn, tp = cm.ravel()
ax6.set_title(f"Konfusionsmatrix – {best_test} (Test-Set)\n"
              f"TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}",
              fontsize=FS_TITLE, fontweight="bold", pad=10)
ax6.set_xlabel("Vom Modell vorhergesagte Klasse", fontsize=FS_AXIS)
ax6.set_ylabel("Tatsächliche Klasse (Ground Truth)", fontsize=FS_AXIS)
ax6.tick_params(labelsize=FS_TICK)

# ── Plot 7: Precision-Recall-Kurven ───────────────────────────────────────
ax7 = fig.add_subplot(gs[3, 0])
for name, res in results_test.items():
    prec, rec, _ = precision_recall_curve(y_te, res["y_proba"])
    ax7.plot(rec, prec, linewidth=2, color=MODEL_COLORS[name], label=name)
ax7.axhline(y_te.mean(), color="gray", linestyle="--", linewidth=1,
            label=f"Baseline-Klassifizierer ({y_te.mean()*100:.1f}%)")
ax7.set_title(f"Precision-Recall-Kurven – Test-Set ({TEST_FROM}–2022)", fontsize=FS_TITLE, fontweight="bold", pad=10)
ax7.set_xlabel("Recall  (= Anteil erkannter echter Verspätungen)", fontsize=FS_AXIS)
ax7.set_ylabel("Precision  (= Genauigkeit der Verspätungswarnungen)", fontsize=FS_AXIS)
ax7.tick_params(labelsize=FS_TICK)
ax7.legend(fontsize=8)

# ── Plot 8: Finale Metriken Vergleich (Test-Set) ──────────────────────────
ax8 = fig.add_subplot(gs[3, 1])
names8  = list(results_test.keys())
auc_v   = [results_test[n]["auc"]      * 100 for n in names8]
acc_v   = [results_test[n]["accuracy"] * 100 for n in names8]
f1_v    = [results_test[n]["f1"]       * 100 for n in names8]
x8 = np.arange(len(names8))
w8 = 0.25
b1 = ax8.bar(x8 - w8, auc_v, w8, label="AUC-ROC (%)",  color=COLORS["primary"],   alpha=0.85, edgecolor="white")
b2 = ax8.bar(x8,       acc_v, w8, label="Accuracy (%)", color=COLORS["ok"],        alpha=0.85, edgecolor="white")
b3 = ax8.bar(x8 + w8,  f1_v,  w8, label="F1-Score (%)", color=COLORS["warn"],      alpha=0.85, edgecolor="white")
for rects in [b1, b2, b3]:
    for rect in rects:
        ax8.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.3,
                 f"{rect.get_height():.1f}", ha="center", va="bottom", fontsize=7)
ax8.set_title(f"Finaler Modellvergleich – Test-Set ({TEST_FROM}–2022)\n"
              f"Höher = besser  |  🏆 {best_test}",
              fontsize=FS_TITLE, fontweight="bold", pad=10)
ax8.set_xlabel("Modell-Algorithmus", fontsize=FS_AXIS)
ax8.set_ylabel("Score (%)", fontsize=FS_AXIS)
ax8.set_xticks(x8)
ax8.set_xticklabels(names8, fontsize=FS_TICK)
ax8.set_ylim(0, 115)
ax8.tick_params(axis="y", labelsize=FS_TICK)
ax8.legend(fontsize=8)

fig.suptitle(
    "Flugverspätungs-Prognose  ·  United Airlines @ JFK  ·  ML-Dashboard",
    fontsize=17, fontweight="bold", y=1.005
)
fig.savefig(OUT_PLOT, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Dashboard gespeichert → {OUT_PLOT}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SPEICHERN
# ─────────────────────────────────────────────────────────────────────────────
lgb_model.save_model(OUT_MODEL)

metrics_out = {
    "split": {
        "train":      f"1987–{TRAIN_UNTIL}",
        "validation": VAL_LABEL,
        "test":       f"{TEST_FROM}–2022",
    },
    "validation_set": {
        n: {k: v for k, v in r.items() if k not in ("model","y_pred","y_proba")}
        for n, r in results_val.items()
    },
    "test_set": {
        n: {k: v for k, v in r.items() if k not in ("model","y_pred","y_proba")}
        for n, r in results_test.items()
    },
    "best_model_validation": best_val,
    "best_model_test":       best_test,
}
with open(OUT_METRICS, "w") as f:
    json.dump(metrics_out, f, indent=2)

print(f"\n{'='*65}")
print("BUSINESS SUMMARY")
print(f"{'='*65}")
for name in results_test:
    r   = results_test[name]
    cm2 = confusion_matrix(y_te, r["y_pred"])
    tn2, fp2, fn2, tp2 = cm2.ravel()
    print(f"""
  {name}
  ├─ AUC-ROC  : {r['auc']*100:.2f}%  (Trennschärfe zwischen Klassen)
  ├─ Accuracy : {r['accuracy']*100:.2f}%  (Anteil korrekt klassifizierter Flüge)
  ├─ F1-Score : {r['f1']*100:.2f}%  (Balance aus Precision & Recall)
  ├─ Threshold: {r['threshold']:.2f}
  ├─ ✅ Erkannte Verspätungen (TP) : {tp2:,} / {int(y_te.sum()):,}
  ├─ ❌ Verpasste Verspätungen (FN) : {fn2:,}
  └─ ⚠️  Fehlalarme (FP)            : {fp2:,}""")

print(f"\n🏆  Empfehlung: {best_test} (höchste AUC auf Test-Set)")
print("✅  Pipeline vollständig abgeschlossen!")

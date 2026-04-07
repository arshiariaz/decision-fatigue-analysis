import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.pipeline import Pipeline
import joblib
import json
import os

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/human_decision_fatigue_dataset.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Load & Inspect ─────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Missing values:\n{df.isnull().sum()}\n")

# ── Feature Engineering ────────────────────────────────────────────────────────
print("Engineering features...")

# Encode Time_of_Day ordinally (morning < afternoon < evening < night)
time_order = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
df["Time_of_Day_Enc"] = df["Time_of_Day"].map(time_order).fillna(1)

# Derived features that a behavioral scientist would care about
df["Sleep_Deficit"]        = np.maximum(0, 7 - df["Sleep_Hours_Last_Night"])       # how far below 7hrs
df["Decision_Density"]     = df["Decisions_Made"] / (df["Hours_Awake"] + 1e-6)     # decisions per hour
df["Cognitive_Pressure"]   = df["Stress_Level_1_10"] * df["Cognitive_Load_Score"]  # combined pressure
df["Fatigue_Risk_Index"]   = (
    df["Hours_Awake"] * 0.3
    + df["Sleep_Deficit"] * 0.4
    + df["Error_Rate"] * 50
    + df["Stress_Level_1_10"] * 0.3
)  # composite risk score inspired by fatigue research

FEATURES = [
    "Hours_Awake",
    "Decisions_Made",
    "Task_Switches",
    "Avg_Decision_Time_sec",
    "Sleep_Hours_Last_Night",
    "Time_of_Day_Enc",
    "Caffeine_Intake_Cups",
    "Stress_Level_1_10",
    "Error_Rate",
    "Cognitive_Load_Score",
    # engineered
    "Sleep_Deficit",
    "Decision_Density",
    "Cognitive_Pressure",
    "Fatigue_Risk_Index",
]

X = df[FEATURES]
y_clf  = df["Fatigue_Level"]            # Low / Moderate / High
y_reg  = df["Decision_Fatigue_Score"]   # continuous 0-100

# ── Encode classification target ──────────────────────────────────────────────
le = LabelEncoder()
le.fit(["Low", "Moderate", "High"])
y_clf_enc = le.transform(y_clf)

# ── Train / Test Split ─────────────────────────────────────────────────────────
X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
    X, y_clf_enc, y_reg, test_size=0.2, random_state=42, stratify=y_clf_enc
)
print(f"  Train size: {len(X_train)} | Test size: {len(X_test)}\n")

# ── Classification Model ───────────────────────────────────────────────────────
print("Training classification model (Fatigue Level)...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, yc_train)

yc_pred = clf.predict(X_test)
clf_accuracy = (yc_pred == yc_test).mean()
print(f"  Accuracy: {clf_accuracy:.4f}")
print(classification_report(yc_test, yc_pred, target_names=le.classes_))

cv_scores = cross_val_score(clf, X, y_clf_enc, cv=5, scoring="accuracy", n_jobs=-1)
print(f"  5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

# ── Regression Model ───────────────────────────────────────────────────────────
print("Training regression model (Fatigue Score)...")
reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
reg.fit(X_train, yr_train)

yr_pred  = reg.predict(X_test)
r2       = r2_score(yr_test, yr_pred)
rmse     = np.sqrt(mean_squared_error(yr_test, yr_pred))
mae      = mean_absolute_error(yr_test, yr_pred)
print(f"  R²:   {r2:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}\n")

# ── Feature Importance ─────────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    "feature":   FEATURES,
    "importance_clf": clf.feature_importances_,
    "importance_reg": reg.feature_importances_,
}).sort_values("importance_clf", ascending=False)

print("Top 10 features (classification):")
print(importance_df[["feature", "importance_clf"]].head(10).to_string(index=False))
print()

# ── Save Artifacts ─────────────────────────────────────────────────────────────
print("Saving models and metadata...")
joblib.dump(clf, f"{MODELS_DIR}/clf_fatigue_level.pkl")
joblib.dump(reg, f"{MODELS_DIR}/reg_fatigue_score.pkl")
joblib.dump(le,  f"{MODELS_DIR}/label_encoder.pkl")

metrics = {
    "clf_accuracy":        round(clf_accuracy, 4),
    "clf_cv_mean":         round(cv_scores.mean(), 4),
    "clf_cv_std":          round(cv_scores.std(), 4),
    "reg_r2":              round(r2, 4),
    "reg_rmse":            round(rmse, 4),
    "reg_mae":             round(mae, 4),
    "n_train":             int(len(X_train)),
    "n_test":              int(len(X_test)),
    "features":            FEATURES,
    "feature_importance":  importance_df[["feature", "importance_clf"]].values.tolist(),
}
with open(f"{MODELS_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ Done. Models saved to /{MODELS_DIR}")
print(f"   clf_fatigue_level.pkl")
print(f"   reg_fatigue_score.pkl")
print(f"   label_encoder.pkl")
print(f"   metrics.json")

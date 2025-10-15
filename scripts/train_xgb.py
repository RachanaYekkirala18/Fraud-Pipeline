import os
import joblib
import pandas as pd
from utils import load_yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

paths = load_yaml("configs/paths.yaml")["local"]
params = load_yaml("configs/params.yaml")

SILVER = paths["silver"]
MODELS = paths["models"]
os.makedirs(MODELS, exist_ok=True)

# Load CSV file instead of Parquet partitions
train_file = f"{SILVER}/train_features.csv"
if not os.path.exists(train_file):
    raise SystemExit("No train_features found. Run spark_etl.py first.")

print(f"Loading {train_file}...")
df = pd.read_csv(train_file)
print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")

label = params["label_col"]
if label not in df.columns:
    raise SystemExit("Train set missing label column. Check ETL.")

# Prepare features
drop_cols = ["TransactionID", "tx_datetime", "dt", label]
X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
y = df[label].astype(int)

print(f"Features shape: {X.shape}")
print(f"Label distribution: {y.value_counts().to_dict()}")

# Simple categorical encoding with pandas.get_dummies (for local demo).
# (For huge data, prefer target/freq encoding or a proper pipeline.)
print("Encoding categorical features...")
X = pd.get_dummies(X, dummy_na=True)

# Fill NaN values to avoid XGBoost/Pandas compatibility issues
X = X.fillna(-999)

print(f"After encoding: {X.shape[1]} features")

print("Splitting train/validation...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=params["train"]["test_size"],
    random_state=params["train"]["random_state"], stratify=y
)

# Convert to numpy arrays to avoid XGBoost/Pandas compatibility issues
X_train = X_train.values if hasattr(X_train, 'values') else X_train
X_val = X_val.values if hasattr(X_val, 'values') else X_val
y_train = y_train.values if hasattr(y_train, 'values') else y_train
y_val = y_val.values if hasattr(y_val, 'values') else y_val

# Handle imbalance
pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
print(f"Positive class weight: {pos_weight:.2f}")

print("Training XGBoost model...")
xgb = XGBClassifier(
    max_depth=params["train"]["xgb"]["max_depth"],
    n_estimators=params["train"]["xgb"]["n_estimators"],
    learning_rate=params["train"]["xgb"]["learning_rate"],
    subsample=params["train"]["xgb"]["subsample"],
    colsample_bytree=params["train"]["xgb"]["colsample_bytree"],
    reg_lambda=params["train"]["xgb"]["reg_lambda"],
    tree_method="auto",  # Changed from "hist" to "auto"
    scale_pos_weight=pos_weight,
    n_jobs=4,
    random_state=params["train"]["random_state"],
    eval_metric="auc"
)

xgb.fit(X_train, y_train)

print("Evaluating on validation set...")
p = xgb.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, p)
aupr = average_precision_score(y_val, p)
print(f"✅ Validation: AUC={auc:.4f} | AUPR={aupr:.4f}")

# Save model
model_path = f"{MODELS}/xgb_ieee.joblib"
# Save column names from the original X DataFrame
joblib.dump({"model": xgb, "columns": X.columns.tolist()}, model_path)
print(f"💾 Saved model → {model_path}")
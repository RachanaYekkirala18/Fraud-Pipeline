import os
import joblib
import pandas as pd
from utils import load_yaml

paths = load_yaml("configs/paths.yaml")["local"]
SILVER = paths["silver"]
GOLD = paths["gold"]
MODELS = paths["models"]
os.makedirs(GOLD, exist_ok=True)

# Load model
bundle_path = f"{MODELS}/xgb_ieee.joblib"
if not os.path.exists(bundle_path):
    raise SystemExit("Model not found. Run train_xgb.py first.")

bundle = joblib.load(bundle_path)
model = bundle["model"]
train_cols = bundle["columns"]
print(f"✓ Loaded model expecting {len(train_cols)} features")

# Load test data from CSV
test_file = f"{SILVER}/test_features.csv"
if not os.path.exists(test_file):
    raise SystemExit("No test_features found. Run spark_etl.py first.")

print(f"Loading {test_file}...")
df = pd.read_csv(test_file, low_memory=False)
print(f"✓ Loaded {len(df):,} test rows")

# Extract TransactionIDs
ids = df["TransactionID"].values if "TransactionID" in df.columns else None

# Prepare features (same columns as training)
drop_cols = ["TransactionID", "tx_datetime", "dt"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

print("Encoding categorical features...")
X = pd.get_dummies(X, dummy_na=True)

# Fill NaN values
X = X.fillna(-999)

print(f"Test data after encoding: {X.shape[1]} features")
print(f"Model expects: {len(train_cols)} features")

# Create a DataFrame with all training columns initialized to 0
print("Aligning features to match training...")
X_aligned = pd.DataFrame(0, index=X.index, columns=train_cols)

# Fill in the values for columns that exist in test data
for col in X.columns:
    if col in X_aligned.columns:
        X_aligned[col] = X[col].values

print(f"✓ Feature alignment complete: {X_aligned.shape}")

# Score
print("Generating predictions...")
# Convert to numpy array for compatibility
X_array = X_aligned.values
scores = model.predict_proba(X_array)[:, 1]

# Create output
out = pd.DataFrame({
    "TransactionID": ids if ids is not None else range(len(scores)),
    "fraud_score": scores
})

# Save as CSV (consistent with our other outputs)
out_path = f"{GOLD}/scored.csv"
out.to_csv(out_path, index=False)
print(f"✅ Wrote {len(out):,} scores → {out_path}")

# Show sample predictions
print("\nSample predictions:")
print(out.head(10))
print(f"\nScore statistics:")
print(out['fraud_score'].describe())
import os
import pandas as pd
from utils import load_yaml

paths = load_yaml("configs/paths.yaml")["local"]
params = load_yaml("configs/params.yaml")
SILVER = paths["silver"]

# Load train features CSV
train_file = f"{SILVER}/train_features.csv"
if not os.path.exists(train_file):
    raise SystemExit("No train_features to validate. Run spark_etl.py first.")

print(f"Loading {train_file} for validation...")
# Load a sample for validation (first 10000 rows for speed)
df = pd.read_csv(train_file, nrows=10000, low_memory=False)
print(f"✓ Loaded {len(df):,} rows for validation")

print("\nRunning data quality checks...")

failed_checks = []
passed_checks = []

# Check 1: TransactionID should not be null
print("  ✓ Checking TransactionID for nulls...")
if df["TransactionID"].isnull().any():
    failed_checks.append("TransactionID contains null values")
else:
    passed_checks.append("TransactionID has no nulls")

# Check 2: TransactionID should be unique
print("  ✓ Checking TransactionID uniqueness...")
if df["TransactionID"].duplicated().any():
    failed_checks.append("TransactionID contains duplicates")
else:
    passed_checks.append("TransactionID is unique")

# Check 3: Amount should be positive
if "amount" in df.columns:
    print("  ✓ Checking amount is positive...")
    if (df["amount"] < 0).any():
        failed_checks.append("amount contains negative values")
    else:
        passed_checks.append("amount is all positive")
    
    if df["amount"].isnull().any():
        failed_checks.append("amount contains null values")
    else:
        passed_checks.append("amount has no nulls")

# Check 4: Fraud label should be binary
label_col = params["label_col"]
if label_col in df.columns:
    print(f"  ✓ Checking {label_col} is binary (0 or 1)...")
    unique_vals = df[label_col].dropna().unique()
    if not all(val in [0, 1] for val in unique_vals):
        failed_checks.append(f"{label_col} contains values other than 0 or 1: {unique_vals}")
    else:
        passed_checks.append(f"{label_col} is binary (0 or 1)")
    
    # Check class balance
    fraud_rate = df[label_col].mean()
    print(f"    Fraud rate: {fraud_rate:.2%}")
    passed_checks.append(f"Fraud rate: {fraud_rate:.2%}")

# Check 5: Time columns exist and are not null
if "tx_datetime" in df.columns:
    print("  ✓ Checking tx_datetime...")
    if df["tx_datetime"].isnull().any():
        failed_checks.append("tx_datetime contains null values")
    else:
        passed_checks.append("tx_datetime has no nulls")

if "dt" in df.columns:
    print("  ✓ Checking date column...")
    if df["dt"].isnull().any():
        failed_checks.append("dt contains null values")
    else:
        passed_checks.append("dt has no nulls")

# Check 6: Verify expected columns exist
print("  ✓ Checking expected columns...")
expected_cols = ["TransactionID", "amount", "hour", "dow", "is_night"]
missing_cols = [col for col in expected_cols if col not in df.columns]
if missing_cols:
    failed_checks.append(f"Missing expected columns: {missing_cols}")
else:
    passed_checks.append("All expected columns present")

# Check 7: Check for unexpected nulls in key features
print("  ✓ Checking for nulls in key features...")
key_features = ["amount", "hour", "dow"]
for col in key_features:
    if col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            failed_checks.append(f"{col} has {null_count} null values")
        else:
            passed_checks.append(f"{col} has no nulls")

# Check 8: Validate numeric ranges
print("  ✓ Checking numeric ranges...")
if "hour" in df.columns:
    if not df["hour"].between(0, 23).all():
        failed_checks.append("hour contains values outside 0-23 range")
    else:
        passed_checks.append("hour is in valid range (0-23)")

if "dow" in df.columns:
    if not df["dow"].between(1, 7).all():
        failed_checks.append("dow contains values outside 1-7 range")
    else:
        passed_checks.append("dow is in valid range (1-7)")

if "is_night" in df.columns:
    if not df["is_night"].isin([0, 1]).all():
        failed_checks.append("is_night contains values other than 0 or 1")
    else:
        passed_checks.append("is_night is binary (0 or 1)")

# Print summary
print("\n" + "="*60)
print("VALIDATION SUMMARY")
print("="*60)
print(f"Total checks: {len(passed_checks) + len(failed_checks)}")
print(f"✅ Passed: {len(passed_checks)}")
print(f"❌ Failed: {len(failed_checks)}")

if passed_checks:
    print("\n✅ PASSED CHECKS:")
    for check in passed_checks:
        print(f"  ✓ {check}")

if failed_checks:
    print("\n❌ FAILED CHECKS:")
    for check in failed_checks:
        print(f"  ✗ {check}")
    print("\n❌ Data quality checks failed")
    raise SystemExit(1)

print("\n✅ All data quality checks passed!")

# Show data sample
print("\n" + "="*60)
print("DATA SAMPLE (first 5 rows)")
print("="*60)
print(df.head())

print("\n" + "="*60)
print("DATA INFO")
print("="*60)
print(f"Shape: {df.shape}")
print(f"Columns ({len(df.columns)}): {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
print(f"\nData types:")
print(df.dtypes.value_counts())
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
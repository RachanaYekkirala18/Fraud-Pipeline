import os
import shutil

# Point Spark to your Hadoop/winutils
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin" + os.pathsep + os.environ.get("PATH", "")
os.environ["PYSPARK_PYTHON"] = r"C:\Users\racha\anaconda3\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\racha\anaconda3\python.exe"

print("winutils resolved to:", shutil.which("winutils"))

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
from utils import load_yaml

# Load configs
paths_all = load_yaml("configs/paths.yaml")
paths = paths_all["local"]
params = load_yaml("configs/params.yaml")

RAW = paths["raw"]
SILVER = paths["silver"]

os.makedirs(SILVER, exist_ok=True)

# Create Spark session
spark = (
    SparkSession.builder
    .appName("fraud-pipeline")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .getOrCreate()
)


def read_join(kind: str):
    """
    kind: 'train' or 'test'
    Returns Spark DF of transaction+identity joined.
    """
    tx = spark.read.csv(f"{RAW}/{kind}_transaction.csv", header=True, inferSchema=True)
    idt = spark.read.csv(f"{RAW}/{kind}_identity.csv", header=True, inferSchema=True)
    df = tx.join(idt, on="TransactionID", how="left")
    return df


def add_time_amount(df):
    """Convert TransactionDT (seconds) to datetime and add amount column"""
    # TransactionDT is seconds from reference date (2017-12-01)
    reference_timestamp = 1512086400  # Unix timestamp for 2017-12-01 00:00:00
    
    df = df.withColumn(
        "tx_datetime",
        (col(params["time_col"]).cast("long") + lit(reference_timestamp)).cast(TimestampType())
    )
    df = df.withColumn("dt", to_date(col("tx_datetime")))
    df = df.withColumn("amount", col(params["amount_col"]).cast("double"))
    return df


def add_basic_flags(df):
    """Add hour, day of week, and night flag features"""
    df = df.withColumn("hour", hour("tx_datetime")) \
           .withColumn("dow", date_format("tx_datetime", "u").cast("int")) \
           .withColumn("is_night", (col("hour").between(0, 5)).cast("int"))
    return df


def safe_exists(df, c):
    """Check if column exists in dataframe"""
    return c in df.columns


def add_rolling_stats(df, key):
    """Add rolling statistics for a given key column"""
    if not safe_exists(df, key):
        return df
    
    w = Window.partitionBy(key).orderBy(col("tx_datetime")) \
             .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    
    df = df.withColumn(f"{key}_cum_txn", count(lit(1)).over(w)) \
           .withColumn(f"{key}_cum_amount", sum("amount").over(w)) \
           .withColumn(f"{key}_mean_amount", avg("amount").over(w)) \
           .withColumn(f"{key}_std_amount", stddev_pop("amount").over(w))
    return df


def amount_zscore(df, key):
    """Calculate z-score for amount based on key statistics"""
    mean_col = f"{key}_mean_amount"
    std_col = f"{key}_std_amount"
    
    if safe_exists(df, mean_col) and safe_exists(df, std_col):
        df = df.withColumn(
            f"amt_z_{key}",
            when(col(std_col).isNull() | (col(std_col) == 0), lit(0.0))
            .otherwise((col("amount") - col(mean_col)) / col(std_col))
        )
    return df


def select_final_columns(df, is_train=True):
    """Select final columns for output"""
    keep = ["TransactionID", "tx_datetime", "dt", "amount", "hour", "dow", "is_night"]
    
    # Add rolling statistics columns
    for key in params["rolling_keys"]:
        if safe_exists(df, key):
            keep += [
                key,
                f"{key}_cum_txn",
                f"{key}_cum_amount",
                f"{key}_mean_amount",
                f"{key}_std_amount",
                f"amt_z_{key}"
            ]
    
    # Add categorical columns
    for c in params["categoricals"]:
        if safe_exists(df, c) and c not in keep:
            keep.append(c)
    
    # Add label column for training data
    if is_train and params["label_col"] in df.columns:
        keep.append(params["label_col"])
    
    # Only keep existing columns
    keep = [c for c in keep if safe_exists(df, c)]
    return df.select(*keep)


def transform(kind: str):
    """Main transformation pipeline"""
    print(f"=== ETL {kind.upper()} ===")
    
    # Read and join data
    df = read_join(kind)
    
    # Rename columns if needed
    if kind == "train" and "isFraud" in df.columns and params["label_col"] != "isFraud":
        df = df.withColumnRenamed("isFraud", params["label_col"])
    if "TransactionAmt" in df.columns and params["amount_col"] != "TransactionAmt":
        df = df.withColumnRenamed("TransactionAmt", params["amount_col"])
    
    # Drop rows with missing critical values
    df = df.dropna(subset=[params["time_col"], params["amount_col"]])
    
    # Add time and amount features
    df = add_time_amount(df)
    df = add_basic_flags(df)
    
    # Add rolling statistics for each key
    for key in params["rolling_keys"]:
        if safe_exists(df, key):
            df = df.withColumn(key, col(key).cast("string"))
            df = add_rolling_stats(df, key)
            df = amount_zscore(df, key)
    
    # Handle missing values in categorical columns
    for c in params["categoricals"]:
        if safe_exists(df, c):
            df = df.withColumn(c, coalesce(col(c).cast("string"), lit("unknown")))
    
    # Select final columns
    out = select_final_columns(df, is_train=(kind == "train"))
    
    # WORKAROUND: Convert to Pandas and save (avoids Hadoop native library issues)
    print(f" Converting to Pandas and saving...")
    output_path = f"{SILVER}/{kind}_features.csv"
    
    # Collect to Pandas DataFrame and save
    pandas_df = out.toPandas()
    pandas_df.to_csv(output_path, index=False)
    
    print(f" ✓ Wrote {kind} features → {output_path}")
    print(f"   Shape: {pandas_df.shape[0]} rows × {pandas_df.shape[1]} columns")
    
    # Clean up
    del pandas_df


# Run transformations
if __name__ == "__main__":
    try:
        transform("train")
        print()
        transform("test")
        print("\n✓ ETL completed successfully!")
    except Exception as e:
        print(f"\n✗ ETL failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        spark.stop()

spark = (
    SparkSession.builder
    .appName("fraud-pipeline")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .config("spark.hadoop.io.nativeio.NativeIO$Windows", "false")  # Add this line
    .getOrCreate()
)
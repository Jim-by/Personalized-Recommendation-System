import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import statistical tools
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import proportion_confint

# Import PySpark modules
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta.tables import DeltaTable

# --- Environment Setup for Local Spark ---
os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['PATH'] += f";{os.environ['HADOOP_HOME']}\\bin"

# --- Constants ---
STATIC_DATA_DIR = "./synthetic_data_stream/static"
ALS_RECS_PATH = "./data/als/user_recommendations"
BPR_RECS_PATH = "./data/bpr/user_recommendations"
USER_MAPPING_PATH = "./data/mart/user_mapping"
ITEM_MAPPING_PATH = "./data/mart/item_mapping"
AB_SAVE_BASE_PATH = "./data/ab_test"

# --- Spark Session Initialization ---
print("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("ABTestSimulator") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()
print("✅ Spark Session initialized.")

# --- Data Loading ---
print("\nLoading static data and model recommendations...")

# Load static user and product data with Pandas
try:
    users_df = pd.read_csv(os.path.join(STATIC_DATA_DIR, "users.csv"))
    products_df = pd.read_csv(os.path.join(STATIC_DATA_DIR, "products.csv"))
except FileNotFoundError as e:
    print(f"❌ Error: Static data not found. Please run the data generation script first. Details: {e}")
    spark.stop()
    sys.exit(1)

# Load ID mappings from Delta tables
user_mapping_df = spark.read.format("delta").load(USER_MAPPING_PATH)
item_mapping_df = spark.read.format("delta").load(ITEM_MAPPING_PATH)
user_idx_to_id = {row.user_idx: row.user_id for row in user_mapping_df.collect()}
item_idx_to_id = {row.item_idx: row.item_id for row in item_mapping_df.collect()}

def load_recs(path, name):
    """Loads recommendations from a Delta table and maps indices back to string IDs."""
    if not DeltaTable.isDeltaTable(spark, path):
        print(f"❌ Recommendations table not found for model '{name}' at: {path}")
        return {}
    
    df = spark.read.format("delta").load(path)
    # Extract item indices from the recommendation struct
    df = df.withColumn("item_indices", F.expr("transform(recommendations, r -> r.item_idx)"))
    
    recs_dict = {}
    for row in df.collect():
        user_id = user_idx_to_id.get(row.user_idx)
        if user_id:
            # Map item indices back to original string IDs
            rec_items = [item_idx_to_id.get(i, f"UNKNOWN_ITEM_{i}") for i in row.item_indices]
            recs_dict[user_id] = rec_items
    print(f"✅ Loaded {len(recs_dict)} recommendations for model '{name}'.")
    return recs_dict

als_recs = load_recs(ALS_RECS_PATH, 'ALS_v1 (Baseline)')
bpr_recs = load_recs(BPR_RECS_PATH, 'BPR_v1 (New Model)')

# --- A/B Test Group Assignment ---
print("\nAssigning users to A/B test groups...")
all_user_ids = users_df["user_id"].astype(str).unique()
np.random.seed(42)  # for reproducibility

# Split users: 40% control, 40% baseline model, 20% new model
groups = np.random.choice(["control", "baseline", "new_model"], p=[0.4, 0.4, 0.2], size=len(all_user_ids))
user_group_mapping = pd.DataFrame({"user_id": all_user_ids, "group": groups})

print("Group sizes:")
print(user_group_mapping['group'].value_counts())

# Assign recommendations based on group
assigned_recs = {}
for _, row in user_group_mapping.iterrows():
    uid = row.user_id
    group = row.group
    if group == "control":
        # Control group gets random popular items
        assigned_recs[uid] = np.random.choice(products_df["item_id"], 10, replace=False).tolist()
    elif group == "baseline":
        # Baseline group gets recommendations from the ALS model
        assigned_recs[uid] = als_recs.get(uid, np.random.choice(products_df["item_id"], 10).tolist())
    else: # new_model
        # Test group gets recommendations from the BPR model
        assigned_recs[uid] = bpr_recs.get(uid, np.random.choice(products_df["item_id"], 10).tolist())

# --- Simulation of User Behavior ---
print("\nSimulating user interactions based on shown recommendations...")

def simulate_interaction(uid, recs, group):
    """Simulates a single user's interaction with a list of recommendations."""
    np.random.seed(abs(hash(uid)) % (2**32)) # User-specific seed
    
    # Simulate showing the top 5 recommendations
    shown_items = recs[:5]
    
    # Simulate a click with a 30% probability
    clicked_item = shown_items[np.random.randint(len(shown_items))] if np.random.rand() < 0.3 else None
    
    # Simulate a purchase with a 20% probability, ONLY if an item was clicked
    purchased_item = clicked_item if clicked_item and np.random.rand() < 0.2 else None
    
    return {
        "user_id": uid,
        "group": group,
        "shown_items": shown_items,
        "clicked_item": clicked_item,
        "purchased_item": purchased_item
    }

# Run the simulation for all users
sim_results = [simulate_interaction(uid, assigned_recs.get(uid, []), grp) for uid, grp in user_group_mapping.values]
simulation_df = pd.DataFrame(sim_results)

# --- Metric Calculation and Statistical Analysis ---
print("\nCalculating metrics and confidence intervals...")

# Calculate CTR and Conversion Rate for each group
metrics = simulation_df.groupby("group").agg(
    conversion_rate=("purchased_item", lambda x: x.notna().mean()),
    ctr=("clicked_item", lambda x: x.notna().mean())
).reset_index()

print("\n--- A/B Test Results ---")
print(metrics)

def get_conversion_ci(df):
    """Calculates Wilson score confidence intervals for conversion rates."""
    records = []
    for group in df['group'].unique():
        group_data = df[df.group == group]
        conversions = group_data["purchased_item"].notna().sum()
        total_users = len(group_data)
        lower, upper = proportion_confint(conversions, total_users, method="wilson")
        records.append({
            "group": group,
            "conversion_rate": conversions / total_users,
            "ci_lower": lower,
            "ci_upper": upper
        })
    return pd.DataFrame(records)

ci_metrics_df = get_conversion_ci(simulation_df)
print("\nConversion Rate with Confidence Intervals:")
print(ci_metrics_df)

# --- Save Results to Delta Lake ---
def save_to_delta(df: pd.DataFrame, path: str, mode="overwrite"):
    """Saves a Pandas DataFrame to a Delta Lake table."""
    print(f"💾 Saving data to Delta table: {path}")
    # Convert Pandas DataFrame to Spark DataFrame
    spark_df = spark.createDataFrame(df)
    (spark_df.write
        .format("delta")
        .mode(mode)
        .option("overwriteSchema", "true") # Useful during development
        .save(path))

print("\nSaving A/B test results to Delta Lake tables...")
os.makedirs(AB_SAVE_BASE_PATH, exist_ok=True)
save_to_delta(simulation_df, os.path.join(AB_SAVE_BASE_PATH, "simulation_events"))
save_to_delta(metrics, os.path.join(AB_SAVE_BASE_PATH, "group_metrics"))
save_to_delta(ci_metrics_df, os.path.join(AB_SAVE_BASE_PATH, "conversion_conf_intervals"))

# Save metadata about this run
meta_info = pd.DataFrame({
    "run_id": [datetime.now().strftime("%Y%m%d_%H%M%S")],
    "click_probability": [0.3],
    "purchase_probability": [0.2],
    "note": ["Automated A/B test simulation with Delta Lake save."]
})
save_to_delta(meta_info, os.path.join(AB_SAVE_BASE_PATH, "ab_runs_metadata"), mode="append")

print("\n✅ A/B test simulation complete. All results are saved in Delta format.")
spark.stop()
print("Spark Session stopped.")

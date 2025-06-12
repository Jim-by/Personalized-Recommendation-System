import os
import sys
import shutil
from datetime import datetime, timedelta
import pathlib  # Used for creating a portable path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from delta.tables import DeltaTable
import pandas as pd

# --- Environment Setup for Local Spark ---
os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['PATH'] = os.environ['PATH'] + ';' + os.environ['HADOOP_HOME'] + r'\bin'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# --- Constants ---
ENRICHED_INTERACTIONS_PATH = "./data/mart/user_item_interactions_enriched"
FEATURE_STORE_BASE_PATH = "./data/feature_store"
USER_FEATURES_PATH = os.path.join(FEATURE_STORE_BASE_PATH, "user_features_delta")
ITEM_FEATURES_PATH = os.path.join(FEATURE_STORE_BASE_PATH, "item_features_delta")

# --- MODIFICATION: Create a portable temporary directory for Spark ---
# This creates a 'tmp/spark_temp' directory inside the project's root folder.
# It makes the script portable and avoids hardcoded paths like 'D:\...'.
# Note: Add 'tmp/' to your .gitignore file!
project_root = pathlib.Path(__file__).parent.resolve()
SPARK_TEMP_DIR = project_root.joinpath("tmp", "spark_temp")
os.makedirs(SPARK_TEMP_DIR, exist_ok=True)
print(f"Using portable temporary directory for Spark: {SPARK_TEMP_DIR}")

def create_spark_session(app_name: str = "FeatureStorePipeline") -> SparkSession:
    """Initializes and returns a Spark Session with Delta Lake support."""
    print("Initializing Spark Session for Feature Store...")
    
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
        .config("spark.hadoop.fs.file.impl.disable.cache", "true")
        .config("spark.sql.warehouse.dir", "file:///C:/tmp/spark-warehouse")
        .config("spark.jars.ivy", os.path.join(os.environ.get('SPARK_HOME', 'C:/spark'), "ivy2"))
        .config("spark.driver.extraJavaOptions", f'-Djava.io.tmpdir="{SPARK_TEMP_DIR}"')
        .config("spark.executor.extraJavaOptions", f'-Djava.io.tmpdir="{SPARK_TEMP_DIR}"')
        .getOrCreate()
    )
    print("Spark Session initialized successfully.")
    return spark

def clean_feature_store():
    """Removes the existing feature store directory to ensure a clean run."""
    if os.path.exists(FEATURE_STORE_BASE_PATH):
        print(f"Found existing Feature Store at: {FEATURE_STORE_BASE_PATH}. Deleting...")
        shutil.rmtree(FEATURE_STORE_BASE_PATH)
        print("Feature Store cleaned successfully.")
    os.makedirs(FEATURE_STORE_BASE_PATH, exist_ok=True)
    print(f"Feature Store directory created: {FEATURE_STORE_BASE_PATH}")

class FeatureStore:
    """
    A class to manage the creation and retrieval of features for users and items.
    It simulates a basic feature store with offline and online access patterns.
    """
    def __init__(self, spark: SparkSession, enriched_interactions_path: str, user_features_path: str, item_features_path: str):
        self.spark = spark
        self.enriched_interactions_path = enriched_interactions_path
        self.user_features_path = user_features_path
        self.item_features_path = item_features_path

    def _load_enriched_interactions(self):
        """Loads the enriched interactions data from its Delta table."""
        print(f"Loading enriched interactions from: {self.enriched_interactions_path}")
        if not DeltaTable.isDeltaTable(self.spark, self.enriched_interactions_path):
            raise FileNotFoundError(
                f"Source Delta table '{self.enriched_interactions_path}' not found. "
                "Please ensure the 'add_temporal_features.py' script has been run."
            )
        return self.spark.read.format("delta").load(self.enriched_interactions_path)

    def _compute_user_features(self, df_interactions, current_processing_date: str):
        """Computes aggregated features for each user."""
        print("Computing user features...")

        user_features = df_interactions.groupBy("user_id").agg(
            F.count("*").alias("total_interactions_count"),
            F.countDistinct("item_id").alias("unique_items_interacted"),
            F.sum(F.when(F.col("interaction_type") == "purchase", 1).otherwise(0)).alias("purchase_count"),
            F.avg(F.col("price_at_interaction")).alias("avg_interaction_price"),
            F.max(F.col("timestamp")).alias("last_interaction_timestamp"),
            F.first("gender").alias("gender"),
            F.first("age").alias("age"),
            F.first("city").alias("city"),
            F.first("device_pref").alias("device_pref"),
            F.first("loyalty_tier").alias("loyalty_tier")
        )
        user_features = user_features.withColumn("processing_date", F.lit(current_processing_date).cast(F.StringType()))
        
        print("User features schema:")
        user_features.printSchema()
        print("Sample user features:")
        user_features.show(3, truncate=False)
        return user_features

    def _compute_item_features(self, df_interactions, current_processing_date: str):
        """Computes aggregated features for each item."""
        print("Computing item features...")
        item_features = df_interactions.groupBy("item_id").agg(
            F.count("*").alias("total_item_interactions"),
            F.countDistinct("user_id").alias("unique_users_interacted_with_item"),
            F.sum(F.when(F.col("interaction_type") == "purchase", 1).otherwise(0)).alias("item_purchase_count"),
            F.avg(F.col("price_at_interaction")).alias("avg_item_interaction_price"),
            F.avg(F.col("product_current_price")).alias("current_avg_product_price"),
            F.max(F.col("timestamp")).alias("last_item_interaction_timestamp"),
            F.first("category").alias("category"),
            F.first("brand").alias("brand"),
            F.first("product_title").alias("product_title")
        )
        item_features = item_features.withColumn("processing_date", F.lit(current_processing_date).cast(F.StringType()))

        print("Item features schema:")
        item_features.printSchema()
        print("Sample item features:")
        item_features.show(3, truncate=False)
        return item_features

    def write_features(self, process_date: datetime):
        """Computes and writes features for a given processing date."""
        current_processing_date_str = process_date.strftime("%Y-%m-%d")
        print(f"\n--- Writing features for date: {current_processing_date_str} ---")

        df_interactions = self._load_enriched_interactions()
        
        user_features_df = self._compute_user_features(df_interactions, current_processing_date_str)
        item_features_df = self._compute_item_features(df_interactions, current_processing_date_str)

        print(f"Writing user features to: {self.user_features_path}")
        user_features_df.write.format("delta").mode("append").option("mergeSchema", "true").save(self.user_features_path)

        print(f"Writing item features to: {self.item_features_path}")
        item_features_df.write.format("delta").mode("append").option("mergeSchema", "true").save(self.item_features_path)

        print(f"Features successfully written for {current_processing_date_str}.")

    def get_features_offline(self, features_type: str, date_str: str = None, version_as_of: int = None):
        """
        Retrieves features for offline use (e.g., model training).
        Can fetch latest, by date, or by version (time travel).
        """
        path = self.user_features_path if features_type == 'user' else self.item_features_path
        id_col = f"{features_type}_id"
        print(f"\nRetrieving OFFLINE features for '{features_type}'...")
        print(f"Path: {path}")

        if not DeltaTable.isDeltaTable(self.spark, path):
            print(f"Warning: Delta table at '{path}' does not exist. Returning empty DataFrame.")
            return self.spark.createDataFrame([], schema=StructType([]))

        reader = self.spark.read.format("delta")
        
        if version_as_of is not None:
            print(f"Querying version: {version_as_of}")
            df = reader.option("versionAsOf", version_as_of).load(path)
        else:
            df = reader.load(path)
        
        # Get the latest version of features for each entity
        window_spec = Window.partitionBy(id_col).orderBy(F.desc("processing_date"))
        df = df.withColumn("row_num", F.row_number().over(window_spec)).filter(F.col("row_num") == 1).drop("row_num")
        
        if date_str:
            print(f"Filtering by processing date: {date_str}")
            df = df.filter(F.col("processing_date") <= date_str)

        print(f"Loaded {df.count()} records for '{features_type}' features.")
        df.show(5, truncate=False)
        return df

    def get_features_online(self, feature_type: str, ids: list):
        """Simulates retrieving the latest features for a list of IDs for online serving."""
        path = self.user_features_path if feature_type == 'user' else self.item_features_path
        id_col = f"{feature_type}_id"
        print(f"\nRetrieving ONLINE features for '{feature_type}' for {len(ids)} ID(s)...")
        print(f"Source path: {path}")

        if not DeltaTable.isDeltaTable(self.spark, path):
            print(f"Warning: Delta table at '{path}' does not exist. Returning empty dictionary.")
            return {}
        
        try:
            # In a real scenario, this would be a highly optimized key-value lookup.
            # Here, we simulate it by reading the latest state from Delta.
            latest_features_df = self.spark.read.format("delta").load(path)
            window_spec = Window.partitionBy(id_col).orderBy(F.desc("processing_date"))
            latest_features_df = latest_features_df.withColumn("row_num", F.row_number().over(window_spec)) \
                .filter(F.col("row_num") == 1) \
                .drop("row_num", "processing_date")
            
            online_df = latest_features_df.filter(F.col(id_col).isin(ids)).toPandas()
            online_features_dict = online_df.set_index(id_col).to_dict('index')
            print(f"Retrieved {len(online_features_dict)} feature sets.")
            return online_features_dict
        except Exception as e:
            print(f"Error during online feature retrieval simulation: {e}")
            return {}

# --- Main Execution Block ---
if __name__ == "__main__":
    hadoop_home = os.environ.get('HADOOP_HOME')
    if not hadoop_home or not os.path.exists(hadoop_home):
        print("Error: HADOOP_HOME is not set or the path does not exist.")
        print("Please download winutils.exe and set the correct path.")
        sys.exit(1)
        
    spark = None
    try:
        spark = create_spark_session()
        clean_feature_store()
        feature_store = FeatureStore(spark, ENRICHED_INTERACTIONS_PATH, USER_FEATURES_PATH, ITEM_FEATURES_PATH)

        # === OFFLINE SCENARIO DEMO: Populating the feature store ===
        print("\n--- DEMO: OFFLINE - Compute and write features for multiple days ---")
        today = datetime.now().date()
        dates_to_process = sorted([today - timedelta(days=i) for i in range(3)])
        for d in dates_to_process:
            feature_store.write_features(d)

        # === OFFLINE SCENARIO DEMO: Reading features for training ===
        print("\n--- DEMO: OFFLINE - Read latest features for model training ---")
        user_features_for_training = feature_store.get_features_offline("user")
        item_features_for_training = feature_store.get_features_offline("item")
        
        # === OFFLINE SCENARIO DEMO: Time Travel ===
        print("\n--- DEMO: OFFLINE - Delta Lake Time Travel ---")
        delta_table_user = DeltaTable.forPath(spark, USER_FEATURES_PATH)
        print("History of the user features table:")
        delta_table_user.history().show(truncate=False)

        last_version = delta_table_user.history().select(F.max("version")).collect()[0][0]
        if last_version is not None and last_version > 0:
            old_version = 0
            print(f"\nReading user features from version {old_version} (time travel)...")
            old_user_features = feature_store.get_features_offline("user", version_as_of=old_version)
        else:
            print("Not enough versions to demonstrate time travel.")

        # === ONLINE SCENARIO DEMO: Reading features for serving ===
        print("\n--- DEMO: ONLINE - Retrieve latest features for serving ---")
        if user_features_for_training.count() > 0:
            sample_user_ids = [row["user_id"] for row in user_features_for_training.limit(3).collect()]
            online_user_features = feature_store.get_features_online("user", sample_user_ids)
            for user_id, features in online_user_features.items():
                print(f"Online features for {user_id}: {features}")
        
        if item_features_for_training.count() > 0:
            sample_item_ids = [row["item_id"] for row in item_features_for_training.limit(3).collect()]
            online_item_features = feature_store.get_features_online("item", sample_item_ids)
            for item_id, features in online_item_features.items():
                print(f"Online features for {item_id}: {features}")

        print("\n✅ Feature Store script demonstrated successfully.")
    
    except FileNotFoundError as fnfe:
        print(f"\nERROR: {fnfe}")
        print("Please ensure you have run 'build_interactions_mart.py' and 'add_temporal_features.py' first.")
    
    except Exception as e:
        print(f"\nA critical error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if spark:
            # --- MODIFICATION: Clean up the portable temp directory ---
            try:
                if os.path.exists(SPARK_TEMP_DIR):
                    shutil.rmtree(SPARK_TEMP_DIR)
                    print(f"Portable temporary directory {SPARK_TEMP_DIR} cleaned up.")
            except Exception as e:
                print(f"Warning: Could not clean up temporary directory {SPARK_TEMP_DIR}: {e}")
            spark.stop()
            print("Spark Session stopped.")

import os
import sys
import shutil
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.sql import functions as F
from delta.tables import DeltaTable

# --- Environment Setup for Local Spark ---
os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['PATH'] = os.environ['PATH'] + ';' + os.environ['HADOOP_HOME'] + r'\bin'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# --- Constants: Paths for Data, Models, and Recommendations ---
INTERACTIONS_PATH = "./data/mart/user_item_interactions_delta"
ALS_MODEL_SAVE_PATH = "./data/als/model"
BPR_MODEL_SAVE_PATH = "./data/bpr/model"  # Simulating a BPR-like model with different ALS params
ALS_RECS_SAVE_PATH = "./data/als/user_recommendations"
BPR_RECS_SAVE_PATH = "./data/bpr/user_recommendations"
USER_MAPPING_PATH = "./data/mart/user_mapping"
ITEM_MAPPING_PATH = "./data/mart/item_mapping"

def create_spark_session(app_name: str) -> SparkSession:
    """Creates and returns a Spark Session with optimized settings for ML tasks."""
    print("Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "6g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    print("Spark Session initialized successfully.")
    return spark

def train_and_save_als_model(
    spark: SparkSession, 
    interactions_df: F.DataFrame, 
    model_path: str, 
    recs_path: str, 
    model_name: str, 
    als_params: dict
):
    """
    Trains an ALS model, saves it, and then generates and saves recommendations.

    Args:
        spark (SparkSession): The active Spark session.
        interactions_df (DataFrame): The DataFrame with user/item interactions and their integer indices.
        model_path (str): The path to save the trained model.
        recs_path (str): The path to save the generated recommendations.
        model_name (str): The name of the model for logging purposes.
        als_params (dict): A dictionary of hyperparameters for the ALS model.
    """
    print(f"\n--- Training and saving {model_name} model ---")

    # Clean up previous artifacts for a fresh run
    if os.path.exists(model_path):
        print(f"Deleting existing model: {model_path}")
        shutil.rmtree(model_path)
    if os.path.exists(recs_path):
        print(f"Deleting existing recommendations: {recs_path}")
        shutil.rmtree(recs_path)

    # Prepare training data: filter for 'purchase' interactions and select necessary columns
    train_df = interactions_df.filter(F.col("interaction_type") == "purchase") \
                              .select("user_idx", "item_idx", "quantity") \
                              .dropna(subset=["user_idx", "item_idx", "quantity"])
    
    if train_df.count() == 0:
        print(f"Error: No data available for training {model_name} after filtering. Skipping.")
        return

    print(f"Number of records for training {model_name}: {train_df.count()}")
    
    # Initialize the ALS model
    als = ALS(
        userCol="user_idx",
        itemCol="item_idx",
        ratingCol="quantity",      # Use 'quantity' as a measure of implicit preference
        implicitPrefs=True,        # Set to True for implicit feedback datasets
        nonnegative=True,          # Ensure embeddings are non-negative
        coldStartStrategy="drop",  # Drop users/items not in the training set
        **als_params               # Apply model-specific hyperparameters
    )

    # Train the model
    model = als.fit(train_df)
    
    # Save the trained model
    model.save(model_path)
    print(f"✅ {model_name} model trained and saved successfully to: {model_path}")

    # --- Generate and Save Recommendations ---
    print(f"Generating top 10 recommendations for all users for {model_name}...")
    # Get a distinct list of all users to generate recommendations for
    all_users_df = interactions_df.select("user_idx").distinct()
    
    # Generate recommendations for each user
    recommendations_df = model.recommendForUserSubset(all_users_df, 10)
    
    # Save recommendations to a Delta Lake table for later evaluation
    (recommendations_df.write
        .format("delta")
        .mode("overwrite")
        .save(recs_path))
    
    print(f"✅ Recommendations for {model_name} saved to: {recs_path}")
    print("Sample of generated recommendations:")
    recommendations_df.show(5, truncate=False)

# --- Main Execution Block ---
if __name__ == "__main__":
    spark = create_spark_session("TrainAndSaveModels")
    
    try:
        print("Loading interaction data from Delta Lake...")
        # Ensure the source data mart exists
        if not DeltaTable.isDeltaTable(spark, INTERACTIONS_PATH):
            print(f"Error: Delta table '{INTERACTIONS_PATH}' not found. "
                  f"Please ensure you have run 'build_interactions_mart.py' and 'add_temporal_features.py'.")
            sys.exit(1)
        interactions_df = spark.read.format("delta").load(INTERACTIONS_PATH)
        
        # --- Indexing and Mapping ---
        # Convert string user_id and item_id to integer indices required by ALS
        print("Indexing user_id and item_id...")
        user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx", handleInvalid="skip")
        item_indexer = StringIndexer(inputCol="item_id", outputCol="item_idx", handleInvalid="skip")

        user_indexer_model = user_indexer.fit(interactions_df)
        indexed_df = user_indexer_model.transform(interactions_df)
        
        item_indexer_model = item_indexer.fit(indexed_df)
        indexed_df = item_indexer_model.transform(indexed_df)

        # Save the mappings from string ID to integer index. This is crucial for interpreting results later.
        print("Saving user_mapping and item_mapping...")
        user_mapping_df = spark.createDataFrame([(int(i), user_id) for i, user_id in enumerate(user_indexer_model.labels)], ["user_idx", "user_id"])
        user_mapping_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(USER_MAPPING_PATH)
        
        item_mapping_df = spark.createDataFrame([(int(i), item_id) for i, item_id in enumerate(item_indexer_model.labels)], ["item_idx", "item_id"])
        item_mapping_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(ITEM_MAPPING_PATH)
        
        print("Mappings saved successfully.")

        # --- Model Training ---
        # Train the first model variant: ALS_v1
        als_v1_params = {
            "rank": 10,
            "maxIter": 5,
            "regParam": 0.1,
            "seed": 42
        }
        train_and_save_als_model(spark, indexed_df, ALS_MODEL_SAVE_PATH, ALS_RECS_SAVE_PATH, "ALS_v1", als_v1_params)

        # Train the second model variant: BPR_v1 (simulated with different ALS hyperparameters)
        als_bpr_v1_params = {
            "rank": 20,       # Increased rank for more complex patterns
            "maxIter": 10,    # More iterations for better convergence
            "regParam": 0.05, # Different regularization
            "alpha": 40.0,    # A common parameter for implicit ALS to adjust confidence
            "seed": 43
        }
        train_and_save_als_model(spark, indexed_df, BPR_MODEL_SAVE_PATH, BPR_RECS_SAVE_PATH, "BPR_v1", als_bpr_v1_params)

    except Exception as e:
        print(f"An error occurred in the main execution block: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'spark' in locals() and spark.getActiveSession():
            spark.stop()
            print("Spark Session stopped.")
            
    print("\n✅ Script train_models.py finished successfully!")

import os
import sys
import math
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Environment Setup for Local Spark ---
os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['PATH'] = os.environ['PATH'] + ';' + os.environ['HADOOP_HOME'] + r'\bin'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Import PySpark modules after setting the environment
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import col, expr, udf
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from delta.tables import DeltaTable

# --- Setup Logging ---
# Configure logger to output to both a file and the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('replay_validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Constants ---
DATA_PATH = "./data/mart/user_item_interactions_delta"
RESULTS_DIR = "./replay_validation_logs"
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, "visualizations")
USER_MAPPING_PATH = "./data/mart/user_mapping"
ITEM_MAPPING_PATH = "./data/mart/item_mapping"
K = 10
BATCH_SIZE_DAYS = 30  # Duration of each test window
MIN_INTERACTIONS = 2  # Minimum interactions for a user/item to be included
MIN_USERS_FOR_TRAINING = 100 # Minimum users required to start a training epoch

# --- Core Functions ---

def create_spark_session(app_name="ReplayValidation"):
    """Creates a Spark Session with Delta Lake support."""
    logger.info("Initializing Spark Session...")
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

def load_and_prepare_data(spark: SparkSession):
    """Loads data, creates integer indices for users/items, and saves mappings."""
    logger.info("Loading data from Delta Lake...")
    raw_df = spark.read.format("delta").load(DATA_PATH)
    
    logger.info("Creating integer indices for users and items...")
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx", handleInvalid="skip")
    item_indexer = StringIndexer(inputCol="item_id", outputCol="item_idx", handleInvalid="skip")
    
    user_indexer_model = user_indexer.fit(raw_df)
    indexer_df = user_indexer_model.transform(raw_df)
    
    item_indexer_model = item_indexer.fit(indexer_df)
    indexer_df = item_indexer_model.transform(indexer_df)
    
    logger.info("Saving user/item mappings to Delta Lake...")
    user_labels = user_indexer_model.labels
    item_labels = item_indexer_model.labels
    
    spark.createDataFrame([(i, uid) for i, uid in enumerate(user_labels)], ["user_idx", "user_id"]) \
        .write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(USER_MAPPING_PATH)
    
    spark.createDataFrame([(i, iid) for i, iid in enumerate(item_labels)], ["item_idx", "item_id"]) \
        .write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(ITEM_MAPPING_PATH)
        
    return indexer_df

def filter_data(data_df: F.DataFrame, min_interactions: int):
    """Filters out users and items with fewer than a specified number of interactions."""
    logger.info(f"Filtering data (min interactions per user/item: {min_interactions})...")
    
    initial_count = data_df.count()
    
    # Filter users
    user_counts = data_df.groupBy("user_idx").count().filter(F.col("count") >= min_interactions)
    filtered_df = data_df.join(user_counts.select("user_idx"), on="user_idx")
    
    # Filter items
    item_counts = filtered_df.groupBy("item_idx").count().filter(F.col("count") >= min_interactions)
    filtered_df = filtered_df.join(item_counts.select("item_idx"), on="item_idx")
    
    filtered_count = filtered_df.count()
    logger.info(f"Records before filtering: {initial_count}")
    logger.info(f"Records after filtering: {filtered_count}")
    logger.info(f"Records removed: {initial_count - filtered_count}")
    
    return filtered_df

def precision_at_k(pred_items, true_items, k):
    """Calculates Precision@K."""
    if not true_items or not pred_items: return 0.0
    pred_k = pred_items[:k]
    return len(set(pred_k) & set(true_items)) / len(pred_k) if pred_k else 0.0

def recall_at_k(pred_items, true_items, k):
    """Calculates Recall@K."""
    if not true_items or not pred_items: return 0.0
    return len(set(pred_items[:k]) & set(true_items)) / len(true_items)

def apk(actual, predicted, k):
    """Calculates Average Precision@K."""
    if not actual or not predicted: return 0.0
    score, hits = 0.0, 0
    actual_set = set(actual)
    for i, p in enumerate(predicted[:k]):
        if p in actual_set:
            hits += 1
            score += hits / (i + 1)
    return score / min(k, len(actual)) if actual else 0.0

def ndcg_at_k(actual, predicted, k):
    """Calculates NDCG@K."""
    if not actual or not predicted: return 0.0
    actual_set = set(actual)
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(predicted[:k]) if item in actual_set)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(actual_set))))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_model(spark, model_name, model, test_df, user_mapping):
    """Evaluates the quality of recommendations from a trained model."""
    try:
        logger.info(f"Evaluating model {model_name}...")
        
        recommendations = model.recommendForAllUsers(K)
        
        # Map user_idx back to user_id for joining with ground truth
        recommendations = recommendations.join(user_mapping, on="user_idx", how="left") \
                                     .withColumn("pred_items", expr("transform(recommendations, x -> x.item_idx)"))

        gt_df = test_df.groupBy("user_id").agg(F.collect_set("item_idx").alias("true_items"))
        joined_df = recommendations.join(gt_df, on="user_id", how="inner").cache()
        
        if joined_df.rdd.isEmpty():
            logger.warning("No matching users found between recommendations and ground truth.")
            return None
        
        udf_precision = udf(lambda p, t: precision_at_k(p, t, K), FloatType())
        udf_recall = udf(lambda p, t: recall_at_k(p, t, K), FloatType())
        udf_mapk = udf(lambda p, t: apk(t, p, K), FloatType())
        udf_ndcg = udf(lambda p, t: ndcg_at_k(t, p, K), FloatType())
        
        metrics_df = joined_df.withColumn("precision", udf_precision("pred_items", "true_items")) \
                              .withColumn("recall", udf_recall("pred_items", "true_items")) \
                              .withColumn("map", udf_mapk("pred_items", "true_items")) \
                              .withColumn("ndcg", udf_ndcg("pred_items", "true_items"))
        
        stats = metrics_df.select(F.avg("precision").alias("precision"), F.avg("recall").alias("recall"),
                                  F.avg("map").alias("map"), F.avg("ndcg").alias("ndcg")).collect()[0]
        
        joined_df.unpersist()
        
        return {
            "model": model_name,
            "MAP@10": round(stats["map"], 4),
            "nDCG@10": round(stats["ndcg"], 4),
            "Precision@10": round(stats["precision"], 4),
            "Recall@10": round(stats["recall"], 4)
        }
    
    except Exception as e:
        logger.error(f"Error during model evaluation for {model_name}: {e}")
        return None

def run_replay_experiment(spark, data_df, model_class, model_params):
    """
    Runs a time-based sliding window experiment (replay validation).
    The model is retrained at each time step on historical data and evaluated on future data.
    """
    logger.info("Starting replay validation experiment...")
    
    data_df = data_df.withColumn("unix_time", F.unix_timestamp(col("timestamp")))
    min_time, max_time = data_df.select(F.min("unix_time"), F.max("unix_time")).collect()[0]
    
    results = []
    current_time = min_time
    user_mapping_df = spark.read.format("delta").load(USER_MAPPING_PATH)
    
    while current_time < max_time:
        epoch_date = datetime.fromtimestamp(current_time)
        logger.info(f"--- Processing Epoch starting at: {epoch_date.strftime('%Y-%m-%d')} ---")
        
        train_df = data_df.filter(col("unix_time") <= current_time)
        
        if train_df.select('user_idx').distinct().count() < MIN_USERS_FOR_TRAINING:
            logger.warning("Not enough distinct users in training data. Skipping this epoch.")
            current_time += BATCH_SIZE_DAYS * 86400  # Move to the next epoch
            continue
        
        next_time = current_time + BATCH_SIZE_DAYS * 86400
        test_df = data_df.filter((col("unix_time") > current_time) & (col("unix_time") <= next_time))
        
        if test_df.rdd.isEmpty():
            logger.warning("No data in the test window. Skipping this epoch.")
            current_time = next_time
            continue
        
        logger.info(f"Training model on {train_df.count()} records...")
        model = model_class(**model_params).fit(train_df)
        
        metrics = evaluate_model(
            spark,
            f"{model_class.__name__}_{epoch_date.strftime('%Y%m%d')}",
            model,
            test_df,
            user_mapping_df
        )
        
        if metrics:
            metrics["timestamp"] = epoch_date.strftime("%Y-%m-%d")
            results.append(metrics)
            logger.info(f"Metrics for epoch {epoch_date.strftime('%Y-%m-%d')}: {metrics}")
        
        current_time = next_time
    
    return pd.DataFrame(results) if results else None

def visualize_results(results_df, title="Experiment Results"):
    """Visualizes the time-series results of the replay validation."""
    if results_df is None or results_df.empty:
        logger.warning("No data to visualize.")
        return

    logger.info(f"Visualizing results: {title}")
    
    plt.figure(figsize=(14, 7))
    metrics_to_plot = ["MAP@10", "nDCG@10", "Precision@10", "Recall@10"]
    
    for metric in metrics_to_plot:
        if metric in results_df.columns:
            plt.plot(results_df["timestamp"], results_df[metric], marker='o', linestyle='-', label=metric)
    
    plt.title(title, fontsize=16)
    plt.xlabel("Time (Start of Training Period)", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plot_path = os.path.join(VISUALIZATION_DIR, f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Chart saved to: {plot_path}")

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # Create output directories
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(VISUALIZATION_DIR, exist_ok=True)
        
        spark = create_spark_session()
        
        # Load, prepare, and filter data
        data_df = load_and_prepare_data(spark)
        filtered_df = filter_data(data_df, MIN_INTERACTIONS)
        
        # Define ALS model parameters
        als_params = {
            "userCol": "user_idx",
            "itemCol": "item_idx",
            "ratingCol": "quantity",
            "implicitPrefs": True,
            "nonnegative": True,
            "coldStartStrategy": "drop",
            "rank": 10,
            "maxIter": 5,
            "regParam": 0.1
        }
        
        # Run the experiment
        replay_results = run_replay_experiment(spark, filtered_df, ALS, als_params)
        
        # Save and visualize results
        if replay_results is not None and not replay_results.empty:
            results_csv_file = os.path.join(RESULTS_DIR, "replay_results.csv")
            replay_results.to_csv(results_csv_file, index=False)
            logger.info(f"Replay validation results saved to {results_csv_file}")
            
            visualize_results(replay_results, "Replay Validation Results")
            
            logger.info("\n--- Final Replay Validation Results ---")
            logger.info("\n" + replay_results.to_string())
        else:
            logger.warning("Experiment did not produce any results.")
        
    except Exception as e:
        logger.error(f"A critical error occurred: {e}", exc_info=True)
        raise
    
    finally:
        if 'spark' in locals() and spark.getActiveSession():
            spark.stop()
            logger.info("Spark Session stopped.")

import os
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Environment Setup for Local Spark ---
os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['PATH'] = os.environ['PATH'] + ';' + os.environ['HADOOP_HOME'] + r'\bin'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Import PySpark modules after setting the environment
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType, StringType
from pyspark.sql.functions import col, udf

# --- Constants ---
K = 10  # The "K" for ranking metrics like Precision@K
GROUND_TRUTH_PATH = "./data/mart/user_item_interactions_delta"
RECOMMENDATION_MODELS = {
    "ALS_v1": "./data/als/user_recommendations",
    "BPR_v1": "./data/bpr/user_recommendations", # Using the second model's recs
}
METRICS_CSV_PATH = "./offline_metrics_logs/metrics.csv"
OUTPUT_DIR = "./offline_metrics_logs"
USER_MAPPING_PATH = "./data/mart/user_mapping"
ITEM_MAPPING_PATH = "./data/mart/item_mapping"

# --- Metric Calculation Functions ---

def precision_at_k(pred_items, true_items, k):
    """Calculates Precision@K."""
    if not true_items or not pred_items:
        return 0.0
    pred_k = pred_items[:k]
    hits = len(set(pred_k) & set(true_items))
    return hits / len(pred_k) if pred_k else 0.0

def recall_at_k(pred_items, true_items, k):
    """Calculates Recall@K."""
    if not true_items or not pred_items:
        return 0.0
    pred_k = pred_items[:k]
    hits = len(set(pred_k) & set(true_items))
    return hits / len(true_items)

def apk(actual, predicted, k):
    """Calculates Average Precision@K (AP@K)."""
    if not actual or not predicted:
        return 0.0
    score, num_hits = 0.0, 0
    actual_set = set(actual)
    for i, p in enumerate(predicted[:k]):
        if p in actual_set and p not in predicted[:i]:
            num_hits += 1
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k) if actual else 0.0

def ndcg_at_k(actual, predicted, k):
    """Calculates Normalized Discounted Cumulative Gain@K (nDCG@K)."""
    if not actual or not predicted:
        return 0.0
    actual_set = set(actual)
    predicted_k = predicted[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(predicted_k) if item in actual_set)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(actual_set), k)))
    return dcg / idcg if idcg > 0 else 0.0

def diversity(predictions):
    """
    Calculates intra-list diversity of all recommendations.
    Accepts a list of pyspark.sql.Row objects.
    """
    # MODIFIED HERE: Access 'pred_items' directly as an attribute of the Row object.
    all_recommended_items = [item for row in predictions for item in row.pred_items if row.pred_items]
    if not all_recommended_items:
        return 0.0
    return len(set(all_recommended_items)) / len(all_recommended_items)

def novelty(predictions, item_popularity_dict):
    """
    Calculates the novelty of recommendations based on item popularity.
    Accepts a list of pyspark.sql.Row objects.
    """
    novelty_scores = []
    for row in predictions:
        # MODIFIED HERE: Access 'pred_items' directly as an attribute.
        rec_ids = row.pred_items
        if not rec_ids:
            continue
        # The less popular an item, the higher its novelty score
        score = sum(1 / math.log2(1 + item_popularity_dict.get(item, 1)) for item in rec_ids) / len(rec_ids)
        novelty_scores.append(score)
    return sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0

# --- Spark and Evaluation Logic ---

def create_spark_session(app_name="OfflineMetricsEval"):
    """Initializes and returns a Spark Session with Delta Lake support."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.sql.debug.maxToStringFields", "1000") \
        .getOrCreate()

def evaluate_model(spark, model_name, model_path, ground_truth_df, item_pop_dict, user_mapping, item_mapping):
    """Evaluates a single recommendation model and returns a dictionary of metrics."""
    print(f"\n{'='*50}\nEvaluating model: {model_name}\n{'='*50}")

    try:
        preds_df = spark.read.format("delta").load(model_path)
        print(f"Loaded {preds_df.count()} recommendations.")
    except Exception as e:
        print(f"Could not load recommendations from {model_path}. Error: {e}")
        return None

    # --- Prepare Recommendations: Map indices back to original IDs ---
    broadcast_item_mapping = spark.sparkContext.broadcast({row.item_idx: row.item_id for row in item_mapping.collect()})
    
    def convert_recommendations(recs):
        if not recs: return []
        mapping = broadcast_item_mapping.value
        return [mapping.get(rec["item_idx"], f"unknown_{rec['item_idx']}") for rec in recs if rec]
    
    convert_udf = udf(convert_recommendations, ArrayType(StringType()))
    
    preds_df = preds_df.join(user_mapping, on="user_idx", how="left") \
                       .withColumn("pred_items", convert_udf("recommendations"))

    # --- Prepare Ground Truth Data ---
    gt_df = ground_truth_df.groupBy("user_id").agg(F.collect_set("item_id").alias("true_items"))

    # --- Join Predictions with Ground Truth ---
    joined_df = preds_df.join(gt_df, on="user_id", how="inner").select("user_id", "pred_items", "true_items")
    joined_df.cache()
    
    if joined_df.rdd.isEmpty():
        print(f"❌ No matching users found between recommendations and test set for model {model_name}.")
        return None
    
    print(f"✅ Found {joined_df.count()} users for evaluation.")
    print("\nSample of joined data for evaluation:")
    joined_df.show(3, truncate=False)

    # --- Calculate Ranking Metrics using UDFs ---
    udf_precision = udf(lambda pred, true: float(precision_at_k(pred, true, K)), FloatType())
    udf_recall = udf(lambda pred, true: float(recall_at_k(pred, true, K)), FloatType())
    udf_mapk = udf(lambda pred, true: float(apk(true, pred, K)), FloatType())
    udf_ndcg = udf(lambda pred, true: float(ndcg_at_k(true, pred, K)), FloatType())

    metrics_df = joined_df.withColumn("precision_at_k", udf_precision("pred_items", "true_items")) \
                          .withColumn("recall_at_k", udf_recall("pred_items", "true_items")) \
                          .withColumn("map_at_k", udf_mapk("pred_items", "true_items")) \
                          .withColumn("ndcg_at_k", udf_ndcg("pred_items", "true_items"))

    stats = metrics_df.select(F.avg("precision_at_k").alias("precision"),
                              F.avg("recall_at_k").alias("recall"),
                              F.avg("map_at_k").alias("map"),
                              F.avg("ndcg_at_k").alias("ndcg")).collect()[0]

    # --- Calculate Catalog and Diversity Metrics ---
    unique_recommended = preds_df.select(F.explode("pred_items").alias("item")).distinct().count()
    unique_catalog = ground_truth_df.select("item_id").distinct().count()
    coverage = unique_recommended / unique_catalog if unique_catalog > 0 else 0.0

    # Collect predictions to driver for diversity/novelty. This works with the modified functions.
    pred_local = joined_df.select("user_id", "pred_items").collect()
    diversity_score = diversity(pred_local)
    novelty_score = novelty(pred_local, item_pop_dict)

    joined_df.unpersist()
    
    result = {
        "model": model_name,
        "MAP@10": round(stats["map"], 4),
        "nDCG@10": round(stats["ndcg"], 4),
        "Precision@10": round(stats["precision"], 4),
        "Recall@10": round(stats["recall"], 4),
        "Coverage": round(coverage, 4),
        "Diversity": round(diversity_score, 4),
        "Novelty": round(novelty_score, 4)
    }
    
    print(f"\n📊 Results for model {model_name}:")
    for metric, value in result.items():
        if metric != "model":
            print(f"  {metric}: {value}")
    
    return result

# --- Main Execution Block ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    spark = create_spark_session()

    # --- Load Mappings and Ground Truth Data ---
    print("Loading user and item mappings...")
    user_mapping = spark.read.format("delta").load(USER_MAPPING_PATH).select("user_idx", "user_id").dropDuplicates()
    item_mapping = spark.read.format("delta").load(ITEM_MAPPING_PATH).select("item_idx", "item_id").dropDuplicates()

    print("Loading ground truth data (purchases)...")
    full_df = spark.read.format("delta").load(GROUND_TRUTH_PATH)
    purchase_df = full_df.filter(col("interaction_type") == "purchase").select("user_id", "item_id").dropna()

    # --- Calculate Item Popularity for Novelty Metric ---
    print("Calculating item popularity...")
    item_popularity = purchase_df.groupBy("item_id").count().toPandas()
    item_pop_dict = dict(zip(item_popularity["item_id"], item_popularity["count"]))

    # --- Evaluate Each Model ---
    all_metrics = []
    for model_name, model_path in RECOMMENDATION_MODELS.items():
        metrics = evaluate_model(spark, model_name, model_path, purchase_df, item_pop_dict, user_mapping, item_mapping)
        if metrics:
            all_metrics.append(metrics)

    spark.stop()

    if not all_metrics:
        print("\n❌ Could not calculate metrics for any model. Exiting.")
        sys.exit(1)

    # --- Save and Visualize Results ---
    df_result = pd.DataFrame(all_metrics)
    df_result.to_csv(METRICS_CSV_PATH, index=False)
    print(f"\n📁 All metrics saved to: {METRICS_CSV_PATH}")

    print(f"\n{'='*80}\n📊 FINAL MODEL COMPARISON RESULTS\n{'='*80}")
    print(df_result.to_string(index=False))

    print("\n📊 Visualizing model comparison...")
    plt.figure(figsize=(15, 8))
    metrics_to_plot = ["MAP@10", "nDCG@10", "Precision@10", "Recall@10", "Coverage", "Diversity", "Novelty"]
    df_melted = df_result.melt(id_vars="model", value_vars=metrics_to_plot,
                               var_name="metric", value_name="value")

    sns.barplot(data=df_melted, x="metric", y="value", hue="model")
    plt.title("Offline Metrics Comparison for Recommendation Models", fontsize=16)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Metric Value")
    plt.xlabel("Metric")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "metrics_comparison.png")
    plt.savefig(plot_path)
    print(f"📈 Comparison chart saved to: {plot_path}")
    plt.show()

    print(f"\n✅ Analysis complete! Results saved in {OUTPUT_DIR}")

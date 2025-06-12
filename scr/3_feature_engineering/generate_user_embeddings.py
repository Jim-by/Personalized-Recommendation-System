import sys
import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    collect_list,
    struct,
    sort_array,
    transform,
    udf
)
from pyspark.sql.types import (
    ArrayType,
    FloatType
)
from pyspark.ml.feature import Word2Vec
from delta.tables import DeltaTable

# --- Environment Setup for Local Spark ---
os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['PATH'] = os.environ['PATH'] + ';' + os.environ['HADOOP_HOME'] + '\\bin'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def create_spark_session(app_name: str) -> SparkSession:
    """Initializes and returns a Spark Session with Delta Lake support."""
    print("Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .getOrCreate()
    print("Spark Session initialized successfully.")
    return spark

def generate_user_embeddings_spark(spark: SparkSession, delta_path: str, vector_size: int = 100, window_size: int = 5, min_count: int = 5):
    """
    Generates user embeddings by training a Word2Vec model on item interaction sequences.

    This function performs the following steps:
    1. Loads interaction data from a Delta table.
    2. Constructs a time-ordered sequence of interacted items for each user.
    3. Trains a Word2Vec model on these sequences to learn item embeddings.
    4. Calculates user embeddings by averaging the embeddings of items they interacted with.
    5. Saves the resulting user embeddings to a new Delta table.

    Args:
        spark (SparkSession): The active Spark session.
        delta_path (str): The path to the source interactions Delta table.
        vector_size (int): The dimensionality of the embedding vectors.
        window_size (int): The context window size for Word2Vec.
        min_count (int): The minimum number of times a word (item) must appear to be included.

    Returns:
        DataFrame or None: The DataFrame containing user embeddings, or None on failure.
    """
    print(f"\nStarting User Embedding generation from Delta table: {delta_path}")

    # --- 1. Load and Validate Data ---
    if not DeltaTable.isDeltaTable(spark, delta_path):
        print(f"Error: Delta table not found at path {delta_path}.")
        return None

    print("Loading data from Delta Lake...")
    interactions_df = spark.read.format("delta").load(delta_path)
    print(f"Loaded {interactions_df.count()} records. Caching...")
    interactions_df.cache()

    # --- 2. Prepare Data for Sequencing ---
    print("Preparing data: selecting columns, filtering, and casting types...")
    base_df = interactions_df.select(
        col("user_id").cast("string"),
        col("item_id").cast("string"),
        col("timestamp").alias("ts")
    )
    base_df = base_df.dropna(subset=["user_id", "item_id", "ts"])
    print(f"{base_df.count()} records remaining after cleanup.")

    # --- 3. Create Item Sequences for Each User ---
    print("Creating ordered sequences of item_ids for each user...")
    # Group by user and collect a list of (timestamp, item_id) structs
    df_with_struct = base_df.withColumn("timed_item", struct(col("ts"), col("item_id")))
    sequenced_df = df_with_struct.groupBy("user_id") \
        .agg(collect_list("timed_item").alias("timed_items_list"))
    
    # Sort the list by timestamp and extract just the item_id to form the sequence
    sequenced_df = sequenced_df.withColumn("sorted_timed_items", sort_array(col("timed_items_list"), asc=True)) \
        .withColumn("sequence", transform(col("sorted_timed_items"), lambda x: x.item_id))

    sequenced_df = sequenced_df.select("user_id", "sequence")
    print("Example of generated sequences:")
    sequenced_df.show(5, truncate=50)

    # --- 4. Train Word2Vec to get Item Embeddings ---
    print("Training Word2Vec model to obtain Item Embeddings...")
    word2vec = Word2Vec(vectorSize=vector_size,
                        windowSize=window_size,
                        minCount=min_count,
                        inputCol="sequence",
                        outputCol="w2v_result") # This output column is temporary
    
    model = word2vec.fit(sequenced_df)
    print("Word2Vec model trained successfully.")
    
    item_embedding_df = model.getVectors()
    print("Example of learned Item Embeddings:")
    item_embedding_df.show(5)

    # --- 5. Create User Embeddings by Averaging Item Embeddings ---
    print("Creating User Embeddings by averaging Item Embeddings...")
    # Collect item vectors to the driver and broadcast them to all executors for efficiency
    item_vectors_dict = {row.word: row.vector.toArray().tolist() for row in item_embedding_df.collect()}
    broadcasted_data = spark.sparkContext.broadcast({
        'vectors': item_vectors_dict,
        'size': vector_size
    })

    # Define a UDF to calculate the user embedding
    def calculate_user_embedding(item_sequence):
        # Retrieve broadcasted data on the executor
        data = broadcasted_data.value
        vectors_dict = data['vectors']
        local_vector_size = data['size']

        user_vectors = []
        if item_sequence is None:
            return [0.0] * local_vector_size

        for item_id in item_sequence:
            if item_id in vectors_dict:
                user_vectors.append(vectors_dict[item_id])

        if user_vectors:
            # Average the vectors of all items in the user's sequence
            return np.mean(user_vectors, axis=0).astype(np.float32).tolist()
        else:
            # Return a zero vector if no valid item embeddings were found
            return [0.0] * local_vector_size

    calculate_user_embedding_udf = udf(calculate_user_embedding, ArrayType(FloatType()))
    user_embeddings_df = sequenced_df.withColumn("user_embedding", calculate_user_embedding_udf(col("sequence")))
    user_embeddings_df = user_embeddings_df.select("user_id", "user_embedding")
    
    # --- 6. Save User Embeddings to Delta Lake ---
    output_path = "./data/mart/user_embeddings_delta"
    print(f"\nSaving User Embeddings to {output_path}...")
    try:
        user_embeddings_df.persist()
        
        count = user_embeddings_df.count()
        print(f"Generated and will be saving {count} User Embeddings.")

        (user_embeddings_df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .save(output_path))
        
        print("User Embeddings saved successfully.")
        
        print("\nExample of saved User Embeddings:")
        user_embeddings_df.show(5, truncate=50)

    except Exception as save_e:
        print(f"An error occurred during count or save: {save_e}")
        return None
    finally:
        user_embeddings_df.unpersist()

    interactions_df.unpersist()
    return user_embeddings_df

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting User Embeddings generation script (PySpark)...")
    
    # Path to the source data mart (use the enriched one for better results if available)
    MART_PATH = "./data/mart/user_item_interactions_enriched"
    
    spark = create_spark_session("UserEmbeddings_Word2Vec_Spark")
    print("Spark Session created.")
    
    try:
        final_user_embeddings = generate_user_embeddings_spark(spark, MART_PATH)
        if final_user_embeddings:
            print("\nUser Embeddings generation completed successfully.")
        else:
            print("\nUser Embeddings generation finished with an error.")
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()
        print("Spark Session stopped.")

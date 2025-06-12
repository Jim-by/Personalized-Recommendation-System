import os
import sys
import shutil
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from delta.tables import DeltaTable

# --- Environment Setup for Local Spark ---
os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['PATH'] = os.environ['PATH'] + ';' + os.environ['HADOOP_HOME'] + '\\bin'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def create_spark_session() -> SparkSession:
    """
    Creates a SparkSession with configurations for Delta Lake and a custom temporary directory.
    This helps manage Spark's temporary files, especially on Windows.
    """
    spark_temp_dir = "D:\\spark_temp"  # Note: Use a path that exists on your system
    if not os.path.exists(spark_temp_dir):
        os.makedirs(spark_temp_dir)
        print(f"Created temporary directory for Spark: {spark_temp_dir}")
        
    return SparkSession.builder \
        .appName("AddTemporalFeatures") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
        .config("spark.jars.ivy", os.path.join(os.environ.get('SPARK_HOME', 'C:/spark'), "ivy2")) \
        .config("spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={spark_temp_dir}") \
        .config("spark.executor.extraJavaOptions", f"-Djava.io.tmpdir={spark_temp_dir}") \
        .getOrCreate()

def add_time_features(spark: SparkSession, input_delta_path: str, output_delta_path: str):
    """
    Adds temporal and user-aggregated features to the interaction data.

    This function reads the main interactions data mart, calculates new features,
    and saves the enriched data to a new Delta table.

    Args:
        spark (SparkSession): The active Spark session.
        input_delta_path (str): Path to the source Delta table.
        output_delta_path (str): Path to the destination Delta table.
    """
    # --- 1. Load Data and Validate ---
    print(f"Loading data from input Delta table: {input_delta_path}")
    if not DeltaTable.isDeltaTable(spark, input_delta_path):
        raise ValueError(f"Input Delta table not found at: {input_delta_path}")
    
    df = spark.read.format("delta").load(input_delta_path)
    
    # Check for required columns before processing
    required_cols = {"user_id", "timestamp", "interaction_type"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Source DataFrame is missing required columns: {missing_cols}")

    # --- 2. Calculate Temporal Features ---
    print("Calculating temporal features...")
    # Define a window partitioned by user and ordered by time to calculate sequential features
    window = Window.partitionBy("user_id").orderBy("timestamp")
    
    df_enriched = df.withColumn(
        "time_since_last_interaction_hr",  # Time in hours
        F.coalesce(
            (F.unix_timestamp("timestamp") - F.lag(F.unix_timestamp("timestamp")).over(window)) / 3600,
            F.lit(0.0)  # First interaction for a user will have 0 time since last
        )
    ).withColumn(
        "day_of_week",  # 0=Sunday, 6=Saturday
        F.dayofweek("timestamp") - 1
    ).withColumn(
        "hour_of_day",
        F.hour("timestamp")
    ).withColumn(
        "is_weekend",
        F.when(F.dayofweek("timestamp").isin([1, 7]), 1).otherwise(0) # Sunday=1, Saturday=7
    ).withColumn(
        "time_of_day_segment",
        F.when(F.hour("timestamp").between(6, 11), "morning")
         .when(F.hour("timestamp").between(12, 17), "afternoon")
         .when(F.hour("timestamp").between(18, 22), "evening")
         .otherwise("night")
    ).withColumn(
        "month",
        F.month("timestamp")
    ).withColumn(
        "day_of_month",
        F.dayofmonth("timestamp")
    )

    # --- 3. Calculate User-Aggregated Features ---
    print("Calculating user-aggregated features...")
    # Pivot interaction types to create counts for each type per user
    user_interaction_counts = df.groupBy("user_id", "interaction_type") \
        .count() \
        .groupBy("user_id") \
        .pivot("interaction_type") \
        .sum("count") \
        .fillna(0)
    
    # Rename pivoted columns for clarity
    for col_name in user_interaction_counts.columns:
        if col_name != "user_id":
            user_interaction_counts = user_interaction_counts.withColumnRenamed(
                col_name, f"user_{col_name}_count"
            )
    
    # Join aggregated features back to the main DataFrame
    df_enriched = df_enriched.join(user_interaction_counts, on="user_id", how="left")
    
    # Calculate total interactions for the user
    count_cols = [c for c in user_interaction_counts.columns if c.startswith("user_") and c.endswith("_count")]
    df_enriched = df_enriched.withColumn(
        "user_total_interactions",
        sum([F.col(c) for c in count_cols])
    )

    # --- 4. Save Enriched Data ---
    print(f"Saving enriched data to Delta table: {output_delta_path}")
    (df_enriched.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(output_delta_path))
    
    # --- 5. Display Results ---
    print("\nThe following features have been added:")
    new_features = [
        "time_since_last_interaction_hr", "day_of_week", "hour_of_day", "is_weekend",
        "time_of_day_segment", "month", "day_of_month", "user_total_interactions"
    ] + count_cols
    
    for feature in new_features:
        if feature in df_enriched.columns:
            print(f"- {feature}")
    
    print(f"\nTotal records in enriched table: {df_enriched.count()}")
    print("\nSample of enriched data:")
    df_enriched.select("user_id", "timestamp", "interaction_type", 
                      "time_since_last_interaction_hr", "hour_of_day", 
                      "is_weekend", "time_of_day_segment", "user_purchase_count").show(5, truncate=False)

# --- Main Execution Block ---
if __name__ == "__main__":
    # Preliminary check for HADOOP_HOME
    hadoop_home = os.environ.get('HADOOP_HOME')
    if not hadoop_home or not os.path.exists(hadoop_home):
        print("ERROR: HADOOP_HOME is not set or the path does not exist.")
        print("Please download winutils.exe and set the HADOOP_HOME environment variable.")
        sys.exit(1)

    spark = None
    try:
        # Initialize Spark Session
        spark = create_spark_session()
        print("Spark Session created successfully.")

        # Define I/O paths
        INPUT_PATH = "./data/mart/user_item_interactions_delta"
        OUTPUT_PATH = "./data/mart/user_item_interactions_enriched"
        
        # Run the feature engineering function
        add_time_features(spark, INPUT_PATH, OUTPUT_PATH)
        print("\nTemporal features were added successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop Spark session and clean up
        if spark:
            spark.stop()
            print("Spark Session stopped.")
            
            # Clean up the temporary directory for Spark
            spark_temp_dir = "D:\\spark_temp"
            try:
                if os.path.exists(spark_temp_dir):
                    shutil.rmtree(spark_temp_dir)
                    print(f"Temporary directory {spark_temp_dir} cleaned successfully.")
            except Exception as e:
                print(f"Could not clean up temporary directory {spark_temp_dir}: {e}")
            
            # Recreate the directory for the next run
            os.makedirs(spark_temp_dir, exist_ok=True)
            print(f"Directory {spark_temp_dir} is ready for the next run.")

    print("\n✅ Script finished successfully!")
    sys.exit(0)

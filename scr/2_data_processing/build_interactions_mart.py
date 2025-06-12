import sys
import os
import shutil
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, to_timestamp, expr, explode, trim
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType, TimestampType, DoubleType
)
from delta.tables import DeltaTable

# --- Environment Setup for Local Spark ---
# Ensure HADOOP_HOME and PYSPARK variables are set for local execution on Windows.
os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['PATH'] = os.environ['PATH'] + ';' + os.environ['HADOOP_HOME'] + r'\bin'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# --- Constants ---
STATIC_DATA_BASE_PATH = "./synthetic_data_stream/static"
RAW_DATA_BASE_PATH = "./synthetic_data_stream/raw"
MART_PATH = "./data/mart/user_item_interactions_delta"

def create_spark_session(app_name: str) -> SparkSession:
    """Initializes and returns a Spark Session with Delta Lake support."""
    print("Initializing Spark Session...")
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0")
        .getOrCreate()
    )
    print("Spark Session initialized successfully.")
    return spark

def is_data_available(raw_data_base_path: str, date_str: str) -> bool:
    """Checks if raw data exists for a specific date."""
    daily_events_dir = os.path.join(raw_data_base_path, "user_events", date_str)
    daily_orders_dir = os.path.join(raw_data_base_path, "orders", date_str)
    daily_search_dir = os.path.join(raw_data_base_path, "search_logs", date_str)
    
    return (
        os.path.exists(daily_events_dir) or
        os.path.exists(daily_orders_dir) or
        os.path.exists(daily_search_dir)
    )

def build_user_item_interactions_mart(
    spark: SparkSession,
    static_data_base_path: str,
    raw_data_base_path: str,
    mart_path: str,
    process_date_str: str
):
    """
    Builds the user-item interactions data mart for a specific date.
    This function ingests, transforms, and enriches raw data.
    """
    print(f"\n--- Starting data mart build for date: {process_date_str} ---")

    # --- 1. Load Static Data (Users and Products) ---
    users_csv_path = os.path.join(static_data_base_path, "users.csv")
    products_csv_path = os.path.join(static_data_base_path, "products.csv")

    users_schema = StructType([
        StructField("user_id", StringType(), True),
        StructField("signup_date", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("city", StringType(), True),
        StructField("device_pref", StringType(), True),
        StructField("total_gmv", FloatType(), True),
        StructField("loyalty_tier", StringType(), True)
    ])
    products_schema = StructType([
        StructField("item_id", StringType(), True),
        StructField("category", StringType(), True),
        StructField("brand", StringType(), True),
        StructField("title", StringType(), True),
        StructField("price", FloatType(), True),
        StructField("stock", IntegerType(), True),
        StructField("image_url", StringType(), True)
    ])

    try:
        users_df = spark.read.csv(users_csv_path, header=True, schema=users_schema, quote='"', escape='"', mode="DROPMALFORMED")
        products_df = spark.read.csv(products_csv_path, header=True, schema=products_schema, quote='"', escape='"', mode="DROPMALFORMED")
        print(f"Loaded {users_df.count()} users and {products_df.count()} products.")
    except Exception as e:
        print(f"Error loading static data: {e}")
        sys.exit(1)

    # --- 2. Load Raw Transactional Data for the Specified Date ---
    daily_events_dir = os.path.join(raw_data_base_path, "user_events", process_date_str)
    daily_orders_dir = os.path.join(raw_data_base_path, "orders", process_date_str)
    daily_search_dir = os.path.join(raw_data_base_path, "search_logs", process_date_str)
    timestamp_format = "yyyy-MM-dd HH:mm:ssXXX"

    user_events_schema = StructType([
        StructField("timestamp", StringType(), True), StructField("user_id", StringType(), True),
        StructField("session_id", StringType(), True), StructField("event_type", StringType(), True),
        StructField("item_id", StringType(), True), StructField("device", StringType(), True),
        StructField("referrer", StringType(), True), StructField("price_at_event", FloatType(), True)
    ])
    orders_schema = StructType([
        StructField("order_id", StringType(), True), StructField("user_id", StringType(), True),
        StructField("order_ts", StringType(), True), StructField("item_ids", StringType(), True),
        StructField("total_amount", FloatType(), True), StructField("promo_code", StringType(), True),
        StructField("payment_method", StringType(), True)
    ])
    search_logs_schema = StructType([
        StructField("timestamp", StringType(), True), StructField("user_id", StringType(), True),
        StructField("query_text", StringType(), True), StructField("result_click_rank", IntegerType(), True),
        StructField("clicked_item_id", StringType(), True)
    ])

    # Load data, creating empty DataFrames if a source is missing to prevent errors
    try:
        user_events_raw = spark.read.csv(daily_events_dir, header=True, schema=user_events_schema, quote='"', escape='"', mode="DROPMALFORMED")
    except Exception:
        user_events_raw = spark.createDataFrame([], schema=user_events_schema)
    try:
        orders_raw = spark.read.csv(daily_orders_dir, header=True, schema=orders_schema, quote='"', escape='"', mode="DROPMALFORMED")
    except Exception:
        orders_raw = spark.createDataFrame([], schema=orders_schema)
    try:
        search_logs_raw = spark.read.csv(daily_search_dir, header=True, schema=search_logs_schema, quote='"', escape='"', mode="DROPMALFORMED")
    except Exception:
        search_logs_raw = spark.createDataFrame([], schema=search_logs_schema)

    # If no data is available for the day, skip processing
    if user_events_raw.count() == 0 and orders_raw.count() == 0 and search_logs_raw.count() == 0:
        print(f"No data available for {process_date_str}. Skipping.")
        return None

    # --- 3. Transform and Unify Data Sources ---
    # Unify each data source into a common schema
    
    # User events (view, add_to_cart, etc.)
    user_events_transformed = (
        user_events_raw.withColumnRenamed("event_type", "interaction_type")
        .withColumn("timestamp", to_timestamp(col("timestamp"), timestamp_format))
        .withColumn("quantity", lit(1))
        .withColumn("price_at_interaction", col("price_at_event"))
        .withColumn("search_query", lit(None).cast(StringType()))
    )
    
    # Orders (exploding item arrays into individual rows)
    orders_transformed = (
        orders_raw.withColumn("item_id_list", expr("from_json(item_ids, 'array<string>')"))
        .withColumn("item_id", explode(col("item_id_list")))
        .withColumn("interaction_type", lit("purchase"))
        .withColumn("timestamp", to_timestamp(col("order_ts"), timestamp_format))
        .withColumn("quantity", lit(1))
        .join(products_df.select(col("item_id").alias("prod_item_id"), col("price").alias("product_current_price")),
              col("item_id") == col("prod_item_id"), "left")
        .withColumn("price_at_interaction", col("product_current_price"))
        .withColumn("session_id", lit(None).cast(StringType()))
        .withColumn("search_query", lit(None).cast(StringType()))
    )
    
    # Search logs
    search_logs_transformed = (
        search_logs_raw.withColumn("interaction_type", lit("search"))
        .withColumn("timestamp", to_timestamp(col("timestamp"), timestamp_format))
        .withColumnRenamed("clicked_item_id", "item_id")
        .withColumn("quantity", lit(0))
        .withColumn("price_at_interaction", lit(0.0).cast(FloatType()))
        .withColumn("session_id", lit(None).cast(StringType()))
        .withColumnRenamed("query_text", "search_query")
    )

    # Union all transformed DataFrames
    common_cols = ["user_id", "item_id", "interaction_type", "timestamp", "quantity", "price_at_interaction",
                   "session_id", "search_query"]

    all_interactions_df = (
        user_events_transformed.select(*common_cols)
        .unionByName(orders_transformed.select(*common_cols), allowMissingColumns=True)
        .unionByName(search_logs_transformed.select(*common_cols), allowMissingColumns=True)
    )
    print(f"Unified {all_interactions_df.count()} interactions.")

    # --- 4. Debugging and Data Cleaning ---
    # A debugging section to ensure join keys are clean and will match
    print("\n--- JOIN Debugging: Cleaning and checking keys ---")
    all_interactions_df = all_interactions_df.withColumn("user_id", trim(col("user_id"))).withColumn("item_id", trim(col("item_id")))
    users_df = users_df.withColumn("user_id", trim(col("user_id")))
    products_df = products_df.withColumn("item_id", trim(col("item_id")))

    print("Sample user_id from interaction stream:")
    all_interactions_df.select("user_id").distinct().show(5, truncate=False)
    print("Sample user_id from user catalog:")
    users_df.select("user_id").distinct().show(5, truncate=False)
    matching_users_count = all_interactions_df.select("user_id").distinct().join(users_df.select("user_id").distinct(), on="user_id", how="inner").count()
    print(f"Number of matching user_ids: {matching_users_count}")

    print("\nSample item_id from interaction stream (not null):")
    all_interactions_df.filter(col("item_id").isNotNull()).select("item_id").distinct().show(5, truncate=False)
    print("Sample item_id from product catalog:")
    products_df.select("item_id").distinct().show(5, truncate=False)
    matching_items_count = all_interactions_df.select("item_id").distinct().join(products_df.select("item_id").distinct(), on="item_id", how="inner").count()
    print(f"Number of matching item_ids: {matching_items_count}")
    print("--- End of JOIN Debugging ---\n")

    # --- 5. Enrich Data and Finalize Schema ---
    final_mart_df = (
        all_interactions_df
        .join(users_df, on="user_id", how="left")
        .join(products_df.withColumnRenamed("price", "product_current_price"), on="item_id", how="left")
        .withColumn("age", col("age").cast(DoubleType()))
        .withColumn("price_at_interaction", col("price_at_interaction").cast(DoubleType()))
        .withColumn("product_current_price", col("product_current_price").cast(DoubleType()))
        .withColumn("processing_date", lit(process_date_str).cast(StringType()))
        .select(
            "user_id", "item_id", "interaction_type", "timestamp",
            "quantity", "price_at_interaction", "session_id",
            "search_query", "gender", "age", "city",
            "device_pref", "loyalty_tier", "category",
            "brand", col("title").alias("product_title"),
            "product_current_price", "processing_date"
        )
    )

    # Fill NULLs with default values to ensure data quality
    final_mart_df = final_mart_df.fillna({
        "session_id": "N/A", "search_query": "",
        "gender": "unknown", "age": -1.0,
        "city": "unknown", "device_pref": "unknown", "loyalty_tier": "new",
        "category": "uncategorized", "brand": "unknown", "product_title": "unknown",
        "price_at_interaction": 0.0, "product_current_price": 0.0
    })
    print(f"Total records after enrichment and NULL filling: {final_mart_df.count()}")
    return final_mart_df

def main():
    """Main function to orchestrate the data mart build process."""
    spark_session = create_spark_session("UserItemInteractionsMartBuilder")

    try:
        # Ask the user for the build mode
        choice = input(
            "What would you like to do?\n"
            "1. Rebuild the entire data mart (full overwrite)\n"
            "2. Add or update data for a specific date (incremental update)\n"
            "Select an option (1/2): "
        )

        if choice == '1':
            # Full rebuild: delete old table and process all available data
            if os.path.exists(MART_PATH):
                print(f"Deleting old table from {MART_PATH}...")
                shutil.rmtree(MART_PATH)
                print("Table successfully deleted.")

            # Find all date partitions with data
            all_possible_dates = sorted([d for d in os.listdir(os.path.join(RAW_DATA_BASE_PATH, "orders"))
                                         if os.path.isdir(os.path.join(RAW_DATA_BASE_PATH, "orders", d))])
            available_dates = [date for date in all_possible_dates if is_data_available(RAW_DATA_BASE_PATH, date)]

            if not available_dates:
                print("No data found for any day. Data mart will not be created.")
                spark_session.stop()
                sys.exit(0)

            # Process each date and union the results
            final_df = None
            for date_str in available_dates:
                df = build_user_item_interactions_mart(
                    spark_session, STATIC_DATA_BASE_PATH, RAW_DATA_BASE_PATH, MART_PATH, date_str
                )
                if df is not None:
                    final_df = df if final_df is None else final_df.unionByName(df)

            # Write the final combined DataFrame to Delta Lake
            if final_df is not None:
                print(f"Writing Delta Lake table to path: {MART_PATH}")
                (final_df.write.format("delta").mode("overwrite")
                 .save(MART_PATH))
                print("✅ User-item interactions mart updated (full overwrite).")
            else:
                print("No data found for any day. Data mart was not created.")

        elif choice == '2':
            # Incremental update for a single date
            process_date_str = input("Enter the date to add/update (YYYY-MM-DD): ")
            try:
                datetime.strptime(process_date_str, "%Y-%m-%d")
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD.")
                spark_session.stop()
                sys.exit(1)

            if not is_data_available(RAW_DATA_BASE_PATH, process_date_str):
                print(f"No data available for {process_date_str}. Update not performed.")
                spark_session.stop()
                sys.exit(0)

            # Build the mart for the specified date
            df = build_user_item_interactions_mart(
                spark_session, STATIC_DATA_BASE_PATH, RAW_DATA_BASE_PATH, MART_PATH, process_date_str
            )

            # Append the new data to the Delta table
            if df is not None:
                print(f"Writing Delta Lake table to path: {MART_PATH}")
                (df.write.format("delta").mode("append")
                 .option("mergeSchema", "true")
                 .save(MART_PATH))
                print("✅ User-item interactions mart updated (incremental update).")
            else:
                print(f"No data available for {process_date_str}. Update not performed.")

        else:
            print("Invalid choice. Exiting.")
            spark_session.stop()
            sys.exit(0)

    finally:
        # Clean up and stop the Spark session
        spark_session.catalog.clearCache()
        spark_session.stop()
        import time; time.sleep(3)
        import gc; gc.collect()
        print("Spark Session stopped.")

if __name__ == "__main__":
    main()

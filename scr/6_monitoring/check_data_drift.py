import os
import sys
import pandas as pd
from datetime import datetime
import warnings

# Suppress common warnings for a cleaner output
warnings.filterwarnings('ignore') 

# --- Import Evidently AI ---
# This block checks for the library and provides a helpful message if it's missing.
try:
    import evidently
    print(f"✅ Evidently AI version {evidently.__version__} imported successfully.")
    
    from evidently.report import Report 
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
        
except ImportError as e:
    print(f"❌ Error importing Evidently AI: {e}")
    print("Please ensure Evidently AI is installed: pip install evidently")
    sys.exit(1)

# --- Constants ---
# NOTE: For a real pipeline, these paths would come from the Feature Store script.
# The dummy data generation below is for standalone demonstration.
REFERENCE_DATA_PATH = "./data/feature_store/user_features_delta/reference.parquet"
CURRENT_DATA_PATH = "./data/feature_store/user_features_delta/current.parquet"
REPORTS_DIR = "./monitoring_reports" # Corrected typo from "raports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_data(reference_path, current_path):
    """Loads the reference and current datasets for drift comparison."""
    print("Loading data for monitoring...")
    try:
        reference = pd.read_parquet(reference_path)
        current = pd.read_parquet(current_path)
        print(f"Reference data shape: {reference.shape}")
        print(f"Current data shape: {current.shape}")
        return reference, current
    except FileNotFoundError as e:
        print(f"❌ Error: Data file not found. Please check paths: {e}")
        sys.exit(1)

def generate_data_drift_report(reference_df, current_df, output_path):
    """
    Generates a comprehensive data drift and quality report using Evidently AI.
    
    Args:
        reference_df (pd.DataFrame): The baseline dataset.
        current_df (pd.DataFrame): The new dataset to compare against the baseline.
        output_path (str): The directory to save the HTML report.
    """
    print("\nGenerating data drift report...")
    
    # Create a report with two main sections: Data Drift and Data Quality
    data_drift_report = Report(metrics=[
        DataDriftPreset(), 
        DataQualityPreset(),
        TargetDriftPreset() # Checks for drift in the target variable, if present
    ])
    
    # Run the report calculation
    data_drift_report.run(reference_data=reference_df, current_data=current_df)
    
    # Save the report to an HTML file
    report_path = os.path.join(output_path, f"data_drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    data_drift_report.save_html(report_path)
    print(f"✅ HTML data drift report saved to: {report_path}")

def monitor_quality_metrics(metrics_csv_path, output_path):
    """
    Visualizes the trend of model quality metrics over time (or different model versions).

    Args:
        metrics_csv_path (str): Path to the CSV file containing model metrics.
        output_path (str): The directory to save the plot.
    """
    print("\nMonitoring recommendation quality metrics...")
    try:
        df = pd.read_csv(metrics_csv_path)
    except FileNotFoundError as e:
        print(f"❌ Error: Quality metrics file not found: {e}")
        return

    # Import plotting libraries locally to keep dependencies clean
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12, 7))
    metrics_to_plot = ["MAP@10", "nDCG@10", "Precision@10", "Recall@10", "Coverage", "Diversity", "Novelty"]
    
    # Melt the dataframe to make it suitable for plotting with seaborn
    df_melted = df.melt(id_vars="model", value_vars=[m for m in metrics_to_plot if m in df.columns],
                        var_name="metric", value_name="value")
                        
    if df_melted.empty:
        print("Warning: No plottable metrics found in the CSV file.")
        return

    sns.lineplot(data=df_melted, x="model", y="value", hue="metric", marker="o", linestyle='--')
    
    plt.title("Trend of Recommendation Quality Metrics", fontsize=16)
    plt.xlabel("Model Version / Run", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Metric")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(output_path, "quality_metrics_trend.png")
    plt.savefig(plot_path)
    print(f"✅ Quality metrics trend chart saved to: {plot_path}")

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # This block creates dummy data if it doesn't exist, making the script runnable standalone for demonstration.
    # In a real pipeline, this data would be produced by the 'create_feature_store.py' script.
    if not os.path.exists(REFERENCE_DATA_PATH):
        print("Dummy reference/current data not found. Creating for demonstration...")
        os.makedirs(os.path.dirname(REFERENCE_DATA_PATH), exist_ok=True)
        # Reference data (e.g., last week's features)
        pd.DataFrame({
            'total_interactions_count': [10, 20, 15, 30, 5],
            'avg_interaction_price': [50.5, 45.2, 60.0, 30.1, 80.8],
            'purchase_count': [1, 3, 2, 5, 0]
        }).to_parquet(REFERENCE_DATA_PATH)
        # Current data (e.g., today's features)
        pd.DataFrame({
            'total_interactions_count': [12, 25, 13, 28, 8],          # Slight drift
            'avg_interaction_price': [70.1, 65.8, 80.2, 55.5, 95.3], # Significant drift
            'purchase_count': [2, 4, 1, 6, 1]
        }).to_parquet(CURRENT_DATA_PATH)
        print("Dummy data created.")

    # This block creates a dummy metrics file if it doesn't exist.
    if not os.path.exists("./offline_metrics_logs/metrics.csv"):
        print("Dummy metrics.csv not found. Creating for demonstration...")
        os.makedirs("./offline_metrics_logs", exist_ok=True)
        pd.DataFrame({
            'model': ['model_v1', 'model_v2', 'model_v3'],
            'MAP@10': [0.15, 0.17, 0.16],
            'nDCG@10': [0.22, 0.25, 0.24],
            'Precision@10': [0.25, 0.28, 0.27],
            'Recall@10': [0.35, 0.38, 0.37],
            'Coverage': [0.4, 0.38, 0.41],
            'Diversity': [0.91, 0.89, 0.92],
            'Novelty': [7.5, 7.3, 7.6]
        }).to_csv("./offline_metrics_logs/metrics.csv", index=False)
        print("Dummy metrics.csv created.")
    
    # --- Run Monitoring Tasks ---
    reference_data, current_data = load_data(REFERENCE_DATA_PATH, CURRENT_DATA_PATH)
    generate_data_drift_report(reference_data, current_data, REPORTS_DIR)
    
    METRICS_CSV_PATH = "./offline_metrics_logs/metrics.csv"
    if os.path.exists(METRICS_CSV_PATH):
        monitor_quality_metrics(METRICS_CSV_PATH, REPORTS_DIR)
    else:
        print(f"Metrics file not found at {METRICS_CSV_PATH}. Skipping quality monitoring.")

    print(f"\n✅ Monitoring of data drift and quality complete. Reports saved in: {REPORTS_DIR}")

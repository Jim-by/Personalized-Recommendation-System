# E-commerce Personalization Engine: A Full-Cycle Recommendation System


**1. Overview**

This project is an end-to-end demonstration of a personalized recommendation system for an e-commerce platform. It showcases the complete MLOps lifecycle, from synthetic data generation and processing to model training, offline evaluation, feature storage, monitoring, and A/B testing.

The primary goal is to simulate a real-world environment and demonstrate the skills required to build, evaluate, and maintain a robust recommendation engine using modern data engineering and machine learning tools.

**2. Key Features**

Synthetic Data Generation: Creates realistic user, product, and interaction data (views, purchases, searches).

ETL with Delta Lake: Processes raw data into a structured and reliable user-item interaction data mart using PySpark and Delta Lake.

Advanced Feature Engineering:

Calculates temporal features (e.g., time since last interaction, day of the week).

Generates user embeddings using Word2Vec on interaction sequences.


Model Training & Recommendation: Implements and trains multiple recommendation models (e.g., ALS-based implicit feedback).

Comprehensive Offline Evaluation:

Compares models using metrics like Precision@K, Recall@K, MAP@K, and nDCG@K.

Analyzes recommendation diversity and catalog coverage.

Performs replay validation to test model performance over time.



Feature Store Implementation: A Delta Lake-based feature store with versioning and support for both offline (training) and online (inference) scenarios.

Monitoring: Uses Evidently AI to generate reports on data drift and quality.

A/B Testing Simulation: Simulates an A/B test to compare the performance of different models in a pseudo-production environment.


**3. Tech Stack**

Data Processing: Apache Spark, Delta Lake, Pandas

Machine Learning: Spark MLlib (ALS, Word2Vec)

Data Quality & Monitoring: Evidently AI

Orchestration & Workflow: Python scripts (can be orchestrated with tools like Airflow or Mage)

Data Visualization: Matplotlib, Seaborn


**4. Project Architecture & Pipeline**

The project is structured as a sequential pipeline. Each step is represented by a script that performs a specific task.



Data Generation (src/1_data_generation/)

Simulates user activity and generates raw CSV files for users, products, events, orders, and search logs.




Data Mart Construction (src/2_data_processing/)

Ingests raw data, cleans it, and builds a central user_item_interactions Delta table, serving as the single source of truth.




Exploratory Data Analysis (EDA) (notebooks/)

Analyzes the data mart to understand distributions, user behavior patterns, and data quality.




Feature Engineering (src/3_feature_engineering/)



Temporal Features: Enriches the data mart with time-based features.

User Embeddings: Creates vector representations of users based on their interaction history.

Feature Store: Calculates and stores user and item features for model training and serving.




Model Training (src/4_training/)

Trains recommendation models (e.g., ALS for implicit feedback) on the processed data and saves the model artifacts and pre-computed recommendations.




Offline Evaluation (src/5_evaluation/)



Metric Calculation: Compares the trained models using various ranking and classification metrics.

Replay Validation: Simulates model performance over historical time windows to ensure stability.




Monitoring (src/6_monitoring/)

Generates reports on data drift in features and tracks the quality of model recommendations over time.




A/B Testing (src/7_ab_testing/)

Runs a simulated A/B test to compare a new model against a baseline and a control group, calculating key business metrics like CTR and Conversion Rate.



**5. Setup and Installation**

Prerequisites

Python 3.9+

Java 8 or 11

Apache Spark (correctly installed and configured)

JAVA_HOME and HADOOP_HOME environment variables set. HADOOP_HOME should point to a directory containing the winutils.exe binary on Windows.

Installation


Clone the repository:


git clone https://github.com/your-username/ecommerce-personalization-engine.git
cd ecommerce-personalization-engine




Create and activate a virtual environment (recommended):


python -m venv venv

*On Windows*

```
.\venv\Scripts\activate
```

*On macOS/Linux*

```
source venv/bin/activate
```




Install the required dependencies:


```
pip install -r requirements.txt
```


**6. How to Run the Pipeline**

The scripts are designed to be run in a specific order. Execute them from the root directory of the project.

* Step 1: Generate synthetic data*
python src/1_data_generation/data_generation.py

* Step 2: Build the main data mart*
python src/2_data_processing/build_interactions_mart.py

* Step 3: Enrich data and create features*
python src/3_feature_engineering/add_temporal_features.py
python src/3_feature_engineering/generate_user_embeddings.py
python src/3_feature_engineering/create_feature_store.py

* Step 4: Train models and generate recommendations
python src/4_training/train_models.py

* Step 5: Evaluate the models
python src/5_evaluation/evaluate_offline_metrics.py
python src/5_evaluation/analyze_recommendations.py
python src/5_evaluation/replay_validation.py

* Step 6: Monitor data drift
python src/6_monitoring/check_data_drift.py

* Step 7: Run A/B test simulation
python src/7_ab_testing/simulate_ab_test.py

**7. Example Results**

Offline Model Comparison
The offline evaluation script compares different models based on ranking metrics. The BPR_v1 model shows a slight improvement in MAP@10 and nDCG@10.

Offline Metrics Comparison

Data Drift Report
The monitoring script generates a detailed HTML report using Evidently AI, highlighting any statistical drift between the reference and current data batches.

**8. Future Work**
This project provides a solid foundation that can be extended in several ways:


Real-time Pipeline: Migrate from batch processing to a real-time stream processing architecture using Kafka and Spark Streaming.

Advanced Models: Implement and evaluate more sophisticated models like LightFM, NCF (Neural Collaborative Filtering), or transformer-based sequential models.

Containerization: Package the entire application using Docker and orchestrate the pipeline with Kubernetes or Airflow.

CI/CD: Implementation a CI/CD pipeline using GitHub Actions to automate testing and deployment.

API for Serving: Develop a REST API (e.g., with FastAPI) to serve recommendations online using the feature store.

**9. License**
This project is licensed under the MIT License. See the LICENSE file for details.



**Contacts**

Author: Uladzimir Manulenka

Email: vlma@tut.by

This project is for demonstration purposes only. All data is synthetic.

# Personalised Recommendation System for E-commerce

## Project Overview

This is a demonstration project for building a personalised recommendation system for an e-commerce platform. The project covers the full data science pipeline: from synthetic data generation and feature engineering to model training, evaluation, monitoring, and A/B testing.

## Tech Stack

- Python, PySpark, Delta Lake
- ClickHouse, PostgreSQL (optional)
- Evidently AI (data & model monitoring)
- Pandas, Matplotlib, Seaborn
- Jupyter Notebook

## Main Stages

1. **Synthetic Data Generation**
2. **User-Item Interaction Mart Construction**
3. **Exploratory Data Analysis (EDA)**
4. **Feature Store (User & Item Features)**
5. **Model Training (ALS, BPR)**
6. **Offline Model Evaluation**
7. **Replay Validation**
8. **Monitoring (Evidently)**
9. **A/B Testing (Simulation)**

## Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

2. **Install dependencies:**


```bash
pip install -r requirements.txt
```


or, if you use conda:

```bash
conda env create -f environment.yml
conda activate your-env-name
```



3. **Generate synthetic data:**

```bash
python scripts/data_generation_1_v2.py
```



4. **Run the pipeline step by step:**



* Build interaction mart:
```bash
python scripts/build_user_item_interactions_mart_2.py
```

* Add temporal features:
```bash
python scripts/add_temporal_features_3.1.py
```

* Train models and generate recommendations:
```bash
python scripts/train_and_save_models_8.1.py
```

* Evaluate and visualize results, etc.


(See the scripts/ folder for all available steps and their descriptions.)



**Contacts**

Author: Uladzimir Manulenka

Email: vlma@tut.by

This project is for demonstration purposes only. All data is synthetic.
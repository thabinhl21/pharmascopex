# PharmaScopeX

## About The Project
PharmaScopeX is a machine learning-powered visualization tool designed to explore drug response patterns across cancel cell lines. Using large-scale pharmacogenomic data from GDSC, we train Random Forest models to predict two widely used drug-response metrics:

- ln(IC50): drug potency
- AUC: drug efficacy

These two predictions are combined into an interactive Pareto-style scatter plot visualization that allows users to explore potency vs efficacy tradeoffs and sensitivity patterns in two complementary modes:

1. Precision Oncology Mode: Discover which drugs are most effective against a chosen cell line
2. Drug Discovery Mode: Discover which cell lines are most sensitive to a chosen drug

## Purpose
This tool is strictly for preclinical, educational, and exploratory purposes. It is not intended to provide medical or clinical recommendations. 

## Installation/Usage

### 1. Clone the repository
```
git clone --recursive <repository-url>
```

### 2. Create a virtual environment and activate the environment
```
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Run application
```
streamlit run visualization/Precision_Oncology.py
```


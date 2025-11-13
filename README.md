<!-- smaller banner -->
<!-- <br />
<p align="center">
  <a href="https://github.com/thabinhl21/pharmascopex">
    <img src="banner.jpg" alt="PharmaScopeX" height="150">
  </a> --> 

<!-- bigger banner -->
![PharmaScopeX](banner.jpg)

<p align="center">
    <strong>An interactive, dual-metric ML engine for preclinical drug sensitivity prediction</strong>
</p>

<p align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" height="28"/>
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" height="28"/>
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white" height="28"/>
    <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" height="28"/>
    <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" height="28"/>
    <img src="https://img.shields.io/badge/Plotly-3C4DBD?logo=plotly&logoColor=white" height="28"/>
</p>

---

## About The Project
PharmaScopeX is a machine learning-powered visualization tool designed to explore drug response patterns across cancel cell lines. Using large-scale pharmacogenomic data from [GDSC](https://www.cancerrxgene.org/), we train Random Forest models to predict two widely used drug-response metrics:

* **ln(IC50)**: drug potency
* **AUC**: drug efficacy

These two predictions are combined into an interactive Pareto-style scatter plot visualization that allows users to explore potency vs efficacy and sensitivity patterns more effectively in two complementary modes:

1. **Precision Oncology Mode**: Discover which drugs are most effective against a chosen cell line
2. **Drug Discovery Mode**: Discover which cell lines are most sensitive to a chosen drug

---

## Purpose
Drug response modeling is complicated, and most tools are hard to interpret without deep ML or biology expertise.  
PharmaScopeX aims to make early-stage exploration **simple, visual, and intuitive**, helping researchers quickly:
* compare drug candidates  
* explore sensitivity trends  
* prototype ideas before diving into wet-lab or computational pipelines 

---

## ⚠️ Limitations

* Does not incorporate genomic features beyond cell line identifiers  
* Simplifies dose–response dynamics using only ln(IC50) & AUC  
* Models trained on limited GDSC subsets  
* Not intended for clinical or diagnostic decision-making. This tool is strictly for preclinical, educational, and exploratory purposes.

---

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


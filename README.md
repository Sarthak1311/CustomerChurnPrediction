# Bank Customer Churn prediciton 
- Practice project implementing MLOps principles to build, track, and deploy a customer churn prediction pipeline.
---
## Table of Contents

- [Project Objective](#project-objective)  
- [Dataset](#dataset)  
- [Methodology / Pipeline](#methodology--pipeline)  
- [Repo Structure](#repo-structure)  
- [Setup & Usage](#setup--usage)  
- [Tools & Technologies](#tools--technologies)  
- [Configuration](#configuration)  
- [How to Contribute](#how-to-contribute)  
- [License](#license)
---

## Project Overview 
Predict which customers are likely to churn (i.e. stop using the service) by building a supervised machine learning pipeline. Along the way, apply MLOps best practices such as reproducibility, versioning, experiment tracking, and configuration management.

---

## Dataset

- The data used is `Customer-Churn-Records.csv`.  
- Basic features include customer demographics, usage metrics, account details, etc.  
- Target variable: whether a customer churned or not.

---

## Methodology / Pipeline

1. **Data Ingestion & Preprocessing** — clean data, handle missing values, encode categorical variables, feature engineering.  
2. **Model Training & Evaluation** — train model(s), validate, measure performance (accuracy, precision, recall, ROC-AUC etc.).  
3. **Experiment Tracking** — track parameters, metrics, artifacts.  
4. **Versioning & Reproducibility** — using tools such as DVC to version data and pipeline stages.  
5. **Configuration Management** — using `params.yaml` and maybe other configuration files.  
6. **Deployment / Monitoring** (if applicable) — if this is extended into production, includes deploying the model and monitoring performance.

---
## Repository Structure
- CustomerChurnPrediction/
    - .dvc/ # DVC directory for versioning data/pipelines
    - dvclive/ # (Optional) For experiment tracking / live metrics
    - src/ # Source code: data processing, model training etc.
    - Customer-Churn-Records.csv # Dataset
    - params.yaml # Hyperparameters / configuration
    - dvc.yaml, dvc.lock # DVC pipeline definitions
    - projectObjective.txt # Project goal / description
    - .gitignore, .dvcignore # Files to ignore for Git / DVC
    - LICENSE # License file

---
### Instructions

1. Clone the repo:
```
git clone https://github.com/Sarthak1311/CustomerChurnPrediction.git
cd CustomerChurnPrediction
```

2. Install required packages
3. Pull the data / set up DVC:
4. Run the pipeline
```
dvc repro
```
---

## Tools & Technologies

- Python – data processing, modeling
- DVC – data & pipeline versioning
- dvclive – experiment tracking / live metrics
- YAML – for configuration (params.yaml)
- (Any ML libraries you used: scikit-learn, pandas, numpy, etc.)
- Git & GitHub for version control


---
## License
This project is licensed under the MIT License.

---
## Authors & Contact
** Sarthak Tyagi **


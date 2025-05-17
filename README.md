# 🚀 Automated ML Pipeline with XGBoost and GitHub Actions

This repository contains an end-to-end machine learning pipeline using **Python**, **XGBoost**, and **GitHub Actions**. The pipeline automates data preprocessing, model training, testing, and artifact handling.

---

## 🔧 What This Project Does

- Loads the **Titanic** dataset
- Drops irrelevant columns (e.g., `Cabin`)
- Fills missing `Age` with **mean** and `Embarked` with **mode**
- Encodes categorical variables
- Trains a **classification model** using **XGBoost**
- Saves the trained model as `model.pkl`
- Includes **unit tests** for preprocessing and accuracy (≥ 80%)

---

## ⚙️ CI/CD with GitHub Actions

On every push or pull request, the GitHub Actions workflow:

1. Sets up Python
2. Installs dependencies
3. Runs tests
4. Trains the model
5. Uploads `model.pkl` as an artifact

---

## 📁 Key Files

- `train.py` – Trains and saves the model  
- `preprocess.py` – Handles data cleaning  
- `test_preprocess.py` – Tests for preprocessing  
- `test_model.py` – Checks model performance  
- `ml_pipeline.yml` – GitHub Actions workflow  
- `requirements.txt` – Project dependencies

---

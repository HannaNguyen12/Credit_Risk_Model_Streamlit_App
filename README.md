# Credit Risk Prediction App (Vietnam Retail Banking)

## Overview

This project develops a **Credit Risk Prediction Model** using a synthetic retail banking dataset representing customers from Vietnam banks. The goal is to identify **high-risk customers** to support better credit decisions and risk management.

The trained model is deployed as an **interactive Streamlit web app**, allowing users to input customer data and receive real-time risk predictions.

**Live Demo:** https://hannanguyencreditriskmodel.streamlit.app
---

## Dataset

This dataset contains **80,000 synthetic retail banking customer records** with realistic financial behavior simulation.

Features include:

- Demographic information  
- Financial data and account balance  
- Service usage and behavioral activity  
- Credit risk labels   

All records are claimed to be fully synthetic and generated programmatically based on real data from Vietnam retail banks

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

Performed EDA and data visualization to understand:

- Feature distributions  
- Risk class imbalance  

### 2. Data Preprocessing

- Encoded categorical features (binary and ordinal encoding)  
- Selected most predictive features using **Mutual Information (MI)**  

Final top 8 features used in the model and app:

- `last_transaction_month` — total transaction amount last month  
- `days_since_last_active`  
- `active_member`  
- `digital_behavior`  
- `nums_service`  
- `balance`  
- `customer_segment`  
- `monthly_ir`  

---

### 3. Model Training and Selection

Compared multiple tree-based models:

- Random Forest  
- XGBoost  
- Extra Trees  

Used **RandomizedSearchCV** for hyperparameter tuning.

Final selected model: **Random Forest Classifier**

---

## Model Performance

Test set performance:

- **ROC-AUC:** 0.924  
- **Precision:** 23.68%  
- **Recall:** 93.01%  

Recall was prioritized to minimize false negatives and ensure high-risk customers are detected.

---

## Deployment

The trained model was exported using joblib and embedded in an interactive web app using Streamlit with:

- Risk probability score
- Interactive interface


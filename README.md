#Streamlit app : https://fraud-pipeline-jcjuux7zbrmjrudqjaejqf.streamlit.app/
# 💳 Financial Fraud Detection Pipeline (IEEE-CIS Dataset)

End-to-end **Data Engineering + Machine Learning pipeline** built with **PySpark**, **XGBoost**, and **Streamlit** to detect fraudulent financial transactions.  
Implements a **bronze → silver → gold** data-lake model, data validation with **Great Expectations**, and feature engineering at scale.

---

## 📘 Project Overview

This project demonstrates a production-style workflow:

1. **Data Ingestion** – merge `train_transaction` and `train_identity` datasets  
2. **ETL with PySpark** – clean, join, and engineer time-aware and rolling features  
3. **Data Quality Checks** – validate schema, nulls, and ranges using Great Expectations  
4. **Model Training** – train an XGBoost model with class-imbalance handling  
5. **Batch Scoring** – generate daily fraud-risk scores (gold layer)  
6. **Visualization** – Streamlit dashboard for fraud trends and high-risk transactions  

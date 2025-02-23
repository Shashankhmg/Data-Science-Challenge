# 🚲 AXA Data Science Challenge - Technical Repository

This repository contains the backend technical analysis, exploratory data analysis (EDA), data preprocessing, and model training for the **AXA Data Science Challenge**. The purpose of this repository is to document the data preparation pipeline and model development process for CitiBike demand prediction and risk assessment using **CityBike** and **NYPD accident data**.

---

## Project Overview

This repository provides a structured approach to:

- **EDA and Data Cleaning**: Understanding the data, handling missing values, feature engineering, and transformation.
- **Model Training & Experimentation**: Building, evaluating, and tuning machine learning models.
- **Logging & Performance Tracking**: Using logs to track training and inference processes.

For the final deployment and interactive visualization, visit the **[Streamlit Application Repository](https://citybikes-nypd-appapp.streamlit.app/)**.

---

## 📂 Repository Structure

```bash
├── data/               # Processed and cleaned datasets (CitiBike + NYPD)
│   ├── raw/            # Original datasets with links below
│   ├── processed/      # Preprocessed datasets for analysis
│   ├── model_data/        # Final cleaned datasets used in modeling
│   ├── encoding/        # encodings required to present the results of model prediction
│
├── notebooks/          # Jupyter Notebooks for EDA & preprocessing
│   ├── 01_eda.ipynb    # Exploratory Data Analysis of CityBike & NYPD datasets
│   ├── 02_cleaning.ipynb  # Data cleaning and preprocessing
│
├── src/                # Source code for modeling and utilities
│   ├── models/         # Scripts for training ML models
│   ├── utils/          # Utility functions for data processing
│
├── heatmaps/           # Risk & accident visualization heatmaps
│
├── main.py             # Main script for training models
├── app.log             # Logs from model training and evaluation
├── README.md           # Documentation
├── .gitignore          # Ignoring large model files & cache
```

---

## 📊 Datasets Used

| Dataset    | Source Link |
|------------|------------|
| **CitiBike (Jan 2025)** | [Download](https://s3.amazonaws.com/tripdata/index.html) |
| **NYPD Accident Data**  | [Download](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data) |

The dataset includes information such as **station details, ride history, accident occurrences & time-based trends**.


## 🛠️ Tech Stack

- **Pandas, NumPy**: Data manipulation and preprocessing
- **Matplotlib, Seaborn**: Data visualization
- **Scikit-learn**: Machine Learning models
- **Joblib**: Model saving and loading
- **FastAPI**: Model serving (separate repository)
- **Logging & Debugging**: `app.log` for tracking model performance

## ⚙️ Model Training & Logging

- All model training logs are stored in **app.log**.
- The models were trained using **Random Forest, Regression Trees, Decision Trees, Linear Regression, K Nearest Neighbours & Naive Bias**.
- Some trained models are missing from this repository due to their **large size**. However, they are hosted on **Hugging Face Hub** or accessible via FastAPI.

## 📂 Additional Resources

- **Deployment Repository**: [Streamlit Interactive App](https://citybikes-nypd-appapp.streamlit.app/)
- **Model Hosting & API**: [FastAPI Model Hosting Github Repo](https://github.com/Shashankhmg/Fastapi-Citybike-Demand-Forecast)
- **Hugging Face Model**: [Download Models]([<huggingface-model-link>](https://huggingface.co/Shashankhmg/citybike-demnd-prediction))

If you are interested in the **backend technical details**, feel free to explore the notebooks and code in this repository. 

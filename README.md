# Airbnb Price Prediction using Machine Learning

This project focuses on predicting Airbnb listing prices in New York City using machine learning techniques. The workflow includes data preprocessing, exploratory data analysis (EDA), clustering, regression modeling, hyperparameter tuning, and deployment using a Flask API.

## Project Overview

The objective of this project is to build predictive models capable of estimating Airbnb prices based on listing characteristics such as location, room type, availability, and review activity.

The project was developed as part of a Statistical Learning Methods (SLM) course project.

## Dataset

* NYC Airbnb Open Data
* 48,000+ records
* Multiple numerical and categorical features

## Machine Learning Workflow

### Data Preprocessing

* Missing value handling
* Feature engineering
* One-hot encoding
* Log transformation of target variable
* Outlier treatment

### Exploratory Data Analysis

* Distribution analysis
* Correlation analysis
* Price and location visualization
* Clustering analysis using K-Means

### Models Implemented

* Linear Regression
* Random Forest Regressor (with hyperparameter tuning)
* XGBoost Regressor

## Final Model

The XGBoost model achieved the best predictive performance and was selected as the final model for deployment.

## Deployment

The final model was exported and integrated into a Flask API capable of generating real-time Airbnb price predictions from JSON input data.

Deployment files included:

* `app.py`
* `requirements.txt`
* `Dockerfile`

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib / Seaborn
* Flask

## Example Prediction

The deployed API successfully generated real-time Airbnb price predictions based on listing characteristics and geographical location.

---

Developed as part of an academic machine learning project.

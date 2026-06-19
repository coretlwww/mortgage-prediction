# Mortgage Loan Amount Prediction

A regression project that predicts the **maximum mortgage loan amount (USD)** an applicant can be approved for, based on demographic and financial features such as income, credit score, employment history, and existing debt.

This project was built as a practical follow-up to Kaggle's [Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) and [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning) courses, applying pipelines, hyperparameter tuning, and model comparison to a real (simulated) dataset end-to-end.

## Dataset

- **Source:** [Mortgage Data](https://www.kaggle.com/datasets/chukwuemeka64/mortgage-data) on Kaggle — a simulated mortgage dataset for loan limit prediction.
- **Size:** 49,990 rows, 14 columns, no missing values.
- **Target variable:** `Max Loan Amount (USD)`
- **Features:** 9 numerical (Age, Annual Income, Credit Score, Employment Years, Interest Rate, Down Payment, Existing Monthly Debt, Loans Repaid) and 5 categorical (Gender, Married, Education, Job, Area).

Since the dataset is simulated, several features show artificial patterns (e.g. capped values at 50 Employment Years and an 850 Credit Score, and near-perfectly balanced categorical groups) — these are called out explicitly in the notebook's EDA and limitations sections.

## Project Structure

```
mortgage-prediction/
├── README.md                  
├── requirements.txt           
├── data/
│   └── mortgage_loan_dataset.csv
├── src/
│   └── utils.py                
├── notebooks/
│   ├── 01_eda.ipynb             
│   └── 02_modeling.ipynb        
```

## Methodology

1. **Data Ingestion & Inspection** — structural audit: shape, dtypes, missing values, duplicates, cardinality.
2. **Exploratory Data Analysis** — distribution analysis (histograms, boxplots), outlier detection via IQR, correlation analysis (full matrix + target-isolated heatmap), and scatter/regression plots for key relationships.
3. **Preprocessing** — a `ColumnTransformer` combining:
   - `SimpleImputer` for numerical features
   - `OneHotEncoder` for low-cardinality categorical features
4. **Train/Validation/Test Split** — 60% / 20% / 20%.
5. **Model Training & Tuning** — three regressors were tuned and compared using validation MAE:
   - `DecisionTreeRegressor` (tuned `max_leaf_nodes`)
   - `RandomForestRegressor` (tuned `n_estimators`)
   - `XGBRegressor` (tuned `n_estimators`, fixed `learning_rate=0.05`)
6. **Final Model** — the best-performing configuration is refit on the combined train+validation data and used to generate predictions on the held-out test set.

## Results

| Model | Best Hyperparameter | Validation MAE |
|---|---|---|
| Decision Tree | `max_leaf_nodes=436` | $44,234 |
| Random Forest | `n_estimators=363` | $19,202 |
| **XGBoost** | `n_estimators=960` | **$14,467** |

XGBoost was selected as the final model, achieving a validation MAE of roughly 2.3% of the median loan amount (~$623,000).

## Key Findings

- **Annual Income** (r = 0.76) and **Credit Score** (r = 0.66) are the strongest individual predictors of the maximum loan amount.
- **Interest Rate** is almost perfectly negatively correlated with Credit Score (r = -0.95), reflecting a rules-based pricing policy.
- **Age** and **Employment Years** are highly collinear (r = 0.97).
- The target variable is right-skewed, with ~0.8% of loans qualifying as statistical outliers (> $1.52M).

## Limitations

- The target variable's right-skew was identified but not corrected with a transformation (e.g. log-transform) — a direction for future iterations.
- Several features show artificial caps and a synthetic, well-balanced distribution, which limits how well the model would generalize to real, unstratified loan applicant data.
- No formal cross-validation was used during hyperparameter tuning — a single validation split was used for all parameter searches.
- A simple baseline model (e.g. predicting the median) was not included for direct comparison.

## Requirements

```
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

## How to Run

```bash
git clone https://github.com/coretlwww/mortgage-prediction.git
cd mortgage-prediction
pip install -r requirements.txt
jupyter notebook mortgage_project.ipynb
```

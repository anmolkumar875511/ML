# ML Projects

A collection of end-to-end machine learning notebooks — from EDA and feature engineering through model comparison, hyperparameter tuning, and blended ensembles. Each project follows a consistent workflow: understand the data, engineer features without leakage, compare multiple model families, tune the strongest candidates, and ship a final model chosen on out-of-fold/held-out performance rather than assumption.

## Projects

### 1. [Customer Churn Prediction](./customer_churn_prediction.ipynb)
Predicts whether a bank customer will exit, evaluated on **F1 score**.

- Full EDA: data types, descriptive statistics, missing values, duplicates, and outliers, each with a documented handling decision (e.g. plausible outliers retained for tree models, missing-indicator flags added where missingness itself carries signal).
- Feature engineering: ratio/interaction features, group-mean deviations, frequency encoding, and out-of-fold target encoding for `last_name`.
- **7 baseline models** compared (Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, XGBoost), with **3 tuned** via `RandomizedSearchCV`.
- **Final model:** a 5-fold out-of-fold blended ensemble of 4 boosting families (HistGradientBoosting, LightGBM, XGBoost, CatBoost), with blend weights and decision threshold optimized directly against F1 — OOF F1 ≈ 0.66, vs. ≈ 0.61–0.62 for the best single-split model.

**Stack:** pandas, scikit-learn, XGBoost, LightGBM, CatBoost

---

### 2. [Flight Ticket Price Prediction](./flight_price_pridiction.ipynb)
Predicts flight ticket prices from airline, route, timing, stops, class, duration, and days-left-to-departure, evaluated on **R²**.

- EDA surfaces a strongly right-skewed, bimodal price distribution driven by the Business/Economy split, plus a corrupted `flight` column (`0.00E+00` spreadsheet artifacts) handled explicitly.
- Baseline pipeline: `StandardScaler` + `OneHotEncoder`/`OrdinalEncoder` inside a single `ColumnTransformer`, fit only on the training split to avoid leakage.
- **10 models** trained across linear, distance-based, and boosted-tree families, with **3 tuned** (XGBoost, LightGBM, CatBoost) — baseline models top out around R² ≈ 0.97.
- **Going further:** leak-safe target (mean) encoding for the high-cardinality `flight` code and an engineered `route` feature, then a blend of the tuned boosted trees — pushes validation **R² past 0.98**.

**Stack:** pandas, scikit-learn, XGBoost, LightGBM, CatBoost

---

### 3. [Heavy Equipment Selling Price Prediction](./heavy_equipment_price_prediction.ipynb)
Predicts resale price of heavy equipment (bulldozers/construction machinery), evaluated on **RMSLE**.

- EDA-driven preprocessing: right-skewed target modeled as `log1p(TargetValue)`; sentinel/placeholder values (e.g. `ManufactureYear < 1900`, zero-hour meters) masked to NaN before feature derivation.
- High-cardinality categoricals (`AssetID`, `Spec_FullDescriptor`, etc.) handled via frequency encoding and leak-safe K-Fold target encoding instead of one-hot.
- Feature engineering guided directly by EDA insights: date-derived features, `AssetAge`, `HoursPerYear`, cyclical month encoding, and target-encoded interaction columns.
- **7 models built in stages:** Ridge baseline, LightGBM, XGBoost, tuned versions of all three, and a final weighted ensemble of the best two by OOF RMSLE.
- Includes a dedicated feature-importance analysis (LightGBM split-gain) and residual/error analysis to identify systematic failure modes beyond the aggregate metric.

**Stack:** pandas, scikit-learn, XGBoost, LightGBM

---

## Common approach across projects

- **Leak-safe preprocessing:** all imputation, scaling, and target-encoding statistics are fit on training data only and applied to validation/test.
- **Multiple model families compared** (linear, distance-based, tree/boosting) before committing to a final approach, with reasoning tied back to EDA insights rather than assumed.
- **Hyperparameter tuning** via `RandomizedSearchCV`/`GridSearchCV` on the strongest candidates.
- **Final model selected empirically** — usually a blended/ensembled model validated with out-of-fold or held-out cross-validation, chosen over single models where it measurably outperforms them.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Matplotlib · Seaborn

## Repo structure

```
.
├── customer_churn_prediction.ipynb
├── flight_price_pridiction.ipynb
├── heavy_equipment_price_prediction.ipynb
└── README.md
```

## Setup

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn
```

Each notebook is self-contained — open it in Jupyter and run top to bottom.

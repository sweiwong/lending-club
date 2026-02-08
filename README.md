# Lending Club Loan Default Prediction

Cost-sensitive binary classification to predict loan defaults, optimizing for financial impact rather than accuracy.

## Key Results

| Metric | Value |
|--------|-------|
| Expected cost per loan | **$684** (vs $1,203 baseline) |
| Default recall | **82%** (catches 4 out of 5 defaults) |
| Per-loan savings | **$519 (43% reduction)** |
| Estimated annual portfolio savings | **$20.1M** |

Logistic Regression was selected over Random Forest and XGBoost because it achieved the lowest expected cost per loan with well-calibrated probabilities.

## Dataset

Lending Club historical loan data — 38,770 approved loans with 23 features. Among borrowers who passed initial underwriting, **14.5% eventually defaulted**.

Source: [Kaggle](https://www.kaggle.com/) · Data file (`loans.csv`) included in this repo.

## Approach

1. **EDA** — explored default rates across loan grades, income levels, and term lengths
2. **Feature engineering** — created loan-to-income ratio (more predictive than raw income), delinquency flags, and categorical encodings (18 final features)
3. **Cost matrix** — defined asymmetric costs: missing a default (FN) costs ~$8,277, rejecting a good loan (FP) costs ~$1,032 (8:1 ratio)
4. **Model comparison** — Logistic Regression, Random Forest, XGBoost evaluated on expected cost, not just AUC
5. **Threshold optimization** — tuned decision threshold to 0.10 to minimize total financial loss
6. **SHAP interpretability** — interest rate and loan-to-income ratio are the top predictors; raw income ranked last

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · matplotlib · seaborn

## How to Run

```bash
git clone https://github.com/YOUR_USERNAME/lending-club-loan-default.git
cd lending-club-loan-default
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
jupyter notebook lending_club_analysis.ipynb
```

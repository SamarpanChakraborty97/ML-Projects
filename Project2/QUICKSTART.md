# Quick Start Guide - Loan Subscription Prediction

## Fast Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
Place `term-deposit-marketing-2020.csv` in the project directory.

### 3. Run the Complete Pipeline

**Option 1: Quick Results**
```bash
# Run the all-features model for best performance
jupyter notebook model_all_features-new-features.ipynb
```

**Option 2: Full Analysis**
```bash
# Step 1: Background variables EDA
jupyter notebook exploratory_data_analysis_background_variables_ver3.ipynb

# Step 2: Outreach variables EDA
jupyter notebook exploratory_data_analysis_outreach_variables_ver3.ipynb

# Step 3: Train model with all features
jupyter notebook model_all_features-new-features.ipynb
```

## Expected Results

After running the model notebook, you should see:
- ✅ Recall: >84%
- ✅ Accuracy: >87%
- ✅ Feature importance plots
- ✅ Confusion matrix visualization
- ✅ Saved model file (.pkl)

## Notebook Execution Order

```
1. exploratory_data_analysis_background_variables_ver3.ipynb
   └─> Generates: term-deposit-marketing-background-variables.csv

2. exploratory_data_analysis_outreach_variables_ver3.ipynb
   └─> Generates: term-deposit-marketing-after-outreach-variables.csv

3. model_all_features-new-features.ipynb (RECOMMENDED)
   └─> Generates: Trained model + predictions
```

## Key Files

| File | Purpose |
|------|---------|
| `*_ver3.ipynb` | Latest version notebooks (use these) |
| `*_ver2.ipynb` | Previous version (for comparison) |
| `model_background_variables*.ipynb` | Baseline model |
| `model_all_features*.ipynb` | Complete model (best results) |

## Troubleshooting

**Problem**: Missing CSV files
```bash
# Ensure you have the original data file:
ls term-deposit-marketing-2020.csv
```

**Problem**: Import errors
```bash
# Reinstall packages
pip install --upgrade -r requirements.txt
```

**Problem**: Low performance
- Try running `model_all_features_extra_different_weights.ipynb` for optimized class weights
- Adjust `scale_pos_weight` parameter in the XGBoost model

## Next Steps

1. Review feature importance plots to understand key predictors
2. Examine partial dependence plots for feature interpretability
3. Adjust hyperparameters if needed
4. Export predictions for business use

## Need Help?

- Check the main README.md for detailed documentation
- Review the EDA notebooks for data insights
- Contact: schakr18@umd.edu

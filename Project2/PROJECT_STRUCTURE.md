# Project Structure and Workflow

## Directory Structure

```
loan-subscription-prediction/
│
├── README.md                                          # Main documentation
├── QUICKSTART.md                                      # Quick start guide
├── requirements.txt                                   # Python dependencies
│
├── Data Files/
│   ├── term-deposit-marketing-2020.csv               # Original dataset
│   ├── term-deposit-marketing-background-variables.csv
│   ├── term-deposit-marketing-after-outreach-variables.csv
│   └── *.csv                                         # Generated feature files
│
├── EDA Notebooks/
│   ├── exploratory_data_analysis_background_variables_ver2.ipynb
│   ├── exploratory_data_analysis_background_variables_ver3.ipynb  ⭐ Latest
│   ├── exploratory_data_analysis_outreach_variables_ver2.ipynb
│   └── exploratory_data_analysis_outreach_variables_ver3.ipynb    ⭐ Latest
│
└── Model Notebooks/
    ├── model_background_variables__1_.ipynb                       # Baseline
    ├── model_all_features-new-features.ipynb                      ⭐ Recommended
    └── model_all_features_extra_different_weights.ipynb           # Optimized
```

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW DATA INPUT                                │
│              term-deposit-marketing-2020.csv                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ├─────────────────────┬──────────────────┐
                         ▼                     ▼                  ▼
         ┌───────────────────────┐ ┌──────────────────────┐      │
         │  Background Variables │ │ Outreach Variables   │      │
         │        EDA            │ │       EDA            │      │
         │  (ver2, ver3)         │ │   (ver2, ver3)       │      │
         └───────────┬───────────┘ └──────────┬───────────┘      │
                     │                        │                  │
                     ├────────────────────────┤                  │
                     ▼                        ▼                  ▼
         ┌──────────────────────────────────────────────────────────┐
         │              FEATURE ENGINEERING                          │
         │   • Age grouping    • Balance categories                 │
         │   • Contact patterns • Campaign features                 │
         │   • Duration analysis • Day categorization               │
         └────────────────────────┬─────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
         ┌────────────────┐ ┌──────────────┐ ┌──────────────────┐
         │   Approach 1   │ │  Approach 2  │ │   Approach 3     │
         │   Background   │ │ All Features │ │  All Features    │
         │   Variables    │ │    Model     │ │  + Weights Opt   │
         │     Only       │ │              │ │                  │
         └────────┬───────┘ └──────┬───────┘ └────────┬─────────┘
                  │                │                   │
                  └────────────────┼───────────────────┘
                                   ▼
                     ┌─────────────────────────────┐
                     │   MODEL TRAINING            │
                     │   • XGBoost Classifier      │
                     │   • SMOTE/Class Weights     │
                     │   • Hyperparameter Tuning   │
                     │   • 5-Fold Cross-Validation │
                     └──────────────┬──────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │   MODEL EVALUATION          │
                     │   • Recall: >84%            │
                     │   • Accuracy: >87%          │
                     │   • Confusion Matrix        │
                     │   • Classification Report   │
                     └──────────────┬──────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │   MODEL INTERPRETATION      │
                     │   • Feature Importance      │
                     │   • Partial Dependence      │
                     │   • SHAP Analysis           │
                     └──────────────┬──────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │   PRODUCTION MODEL          │
                     │   • Saved .pkl file         │
                     │   • Ready for deployment    │
                     └─────────────────────────────┘
```

## Version History

### Version 3 (Latest) ⭐
- **EDA**: `*_ver3.ipynb` notebooks
- Improved feature engineering
- Enhanced visualization
- Better categorical handling

### Version 2
- **EDA**: `*_ver2.ipynb` notebooks
- Initial feature engineering
- Basic exploratory analysis

### Model Versions

| Model | Features Used | Special Focus | Best For |
|-------|--------------|---------------|----------|
| `model_background_variables__1_` | Demographics only | Baseline performance | Quick predictions without campaign data |
| `model_all_features-new-features` | All features | Comprehensive approach | Best overall performance |
| `model_all_features_extra_different_weights` | All features | Class weight optimization | Handling severe imbalance |

## Data Flow

```
Original Data (40,000 records)
    │
    ├─> Background Variables Processing
    │   ├─> Age → Age Groups (3 categories)
    │   ├─> Balance → Balance Groups (4 categories)
    │   └─> Job, Marital, Education → Encoded
    │
    ├─> Outreach Variables Processing
    │   ├─> Campaign → Contact frequency features
    │   ├─> Duration → Duration categories
    │   ├─> Day/Month → Temporal features
    │   └─> Previous campaigns → Historical features
    │
    └─> Feature Selection
        └─> Combined Feature Set
            │
            ├─> Train Set (80% / ~32,000 records)
            │   └─> SMOTE Application → Balanced Training Data
            │
            └─> Test Set (20% / ~8,000 records)
                └─> Model Evaluation
```

## Feature Categories

### Background Variables (Demographic)
```
Demographics
├─ age                    → age_group (categorical)
├─ job                    → encoded
├─ marital               → encoded
├─ education             → encoded
│
Account Information
├─ balance               → balance_group (categorical)
├─ housing               → encoded
├─ loan                  → encoded
└─ default               → encoded
```

### Outreach Variables (Campaign)
```
Contact Information
├─ contact               → contact type
├─ day                   → day_category
└─ month                 → month encoded

Campaign Metrics
├─ duration              → key predictor
├─ campaign              → contact frequency
├─ pdays                 → days since last contact
├─ previous              → previous contacts
└─ poutcome             → previous campaign outcome
```

## Model Pipeline

```python
# Simplified pipeline structure
1. Data Loading → pd.read_csv()
2. Feature Engineering → custom transformations
3. Train-Test Split → train_test_split(test_size=0.2)
4. Data Balancing → SMOTE() or scale_pos_weight
5. Model Training → XGBClassifier()
6. Hyperparameter Tuning → RandomizedSearchCV()
7. Evaluation → confusion_matrix(), classification_report()
8. Interpretation → feature_importances_, partial_dependence()
9. Model Saving → joblib.dump()
```

## Performance Metrics Priority

```
Primary Metric: RECALL (>84%) 
    ↓
    Identifies maximum subscribers
    Critical for business ROI

Secondary Metric: ACCURACY (>87%)
    ↓
    Overall model reliability
    
Tertiary Metrics:
    • Precision
    • F1-Score
    • ROC-AUC
```

## Best Practices

1. **Always use ver3 notebooks** for EDA (most updated)
2. **Run EDA before modeling** to understand data distributions
3. **Use all-features model** for production (best performance)
4. **Monitor recall primarily** as it's the business-critical metric
5. **Save models with timestamps** for version control
6. **Document hyperparameter changes** for reproducibility

## Common Customization Points

1. **Feature Engineering**: Modify in EDA notebooks
   - Adjust age/balance group boundaries
   - Add new derived features

2. **Class Imbalance Handling**: Change in model notebooks
   - SMOTE parameters
   - Class weights
   - Sampling strategies

3. **Hyperparameters**: Adjust in RandomizedSearchCV
   - learning_rate
   - max_depth
   - n_estimators

4. **Evaluation Focus**: Modify based on business needs
   - Recall vs Precision trade-off
   - Threshold adjustment

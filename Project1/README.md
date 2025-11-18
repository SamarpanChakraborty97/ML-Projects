# Customer Satisfaction Prediction

A machine learning project to predict customer satisfaction for a food delivery service based on various service quality metrics. This project implements and compares multiple classification algorithms to identify happy and unhappy customers, achieving over 85% accuracy.

## 📋 Project Overview

This project analyzes customer survey data from ACME to predict customer satisfaction levels. Using various machine learning techniques including Random Forest and XGBoost, the model helps identify key factors influencing customer happiness and provides actionable insights for service improvement.

### Key Features

- **Binary Classification**: Predicts whether a customer is happy (1) or unhappy (0)
- **Feature Engineering**: Comprehensive analysis of 6 service quality metrics
- **Model Comparison**: Implementation of multiple ML algorithms including:
  - Random Forest Classifier
  - XGBoost (Boosted Trees)
  - Support Vector Machines (SVM)
- **Imbalanced Data Handling**: SMOTE technique for balanced training
- **Interpretability**: Feature importance analysis and partial dependency plots
- **High Accuracy**: Achieved 85%+ accuracy and 84%+ recall rate

## 📊 Dataset Description

**Dataset**: ACME-HappinessSurvey2020.csv
- **Total Samples**: 126 customer responses
- **Features**: 6 ordinal variables (ratings from 1-5)
- **Target Variable**: Binary (Happy/Unhappy)

### Features Explained

| Feature | Description | Rating Scale |
|---------|-------------|--------------|
| **X1** | Timely Order Delivery | 1-5 |
| **X2** | Expected Content of Order Delivered | 1-5 |
| **X3** | Adequacy of Ingredients to Order | 1-5 |
| **X4** | Value for Money | 1-5 |
| **X5** | Courier Satisfaction | 1-5 |
| **X6** | Easy Application Interface | 1-5 |
| **Y** | Customer Satisfaction (Target) | 0 (Unhappy) / 1 (Happy) |

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SamarpanChakraborty97/ML-Projects.git
cd ML-Projects/Project1
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

### Required Libraries

#### Core Dependencies
- **numpy** (1.26.4) - Numerical computing
- **pandas** (2.1.4) - Data manipulation and analysis
- **scipy** (1.11.4) - Scientific computing

#### Machine Learning
- **scikit-learn** (1.3.2) - Machine learning algorithms
- **xgboost** (2.0.3) - Gradient boosting framework

#### Deep Learning
- **tensorflow** (2.15.0) - Deep learning framework
- **keras** (2.15.0) - Neural network API

#### Optimization & Statistics
- **statsmodels** (0.14.1) - Statistical modeling
- **optuna** (3.5.0) - Hyperparameter optimization
- **lazypredict** (0.2.12) - AutoML utility

#### Visualization
- **seaborn** (0.13.2) - Statistical data visualization
- **matplotlib** (3.8.2) - Plotting library

## 💻 How to Run

### Option 1: Run the Main Notebook

```bash
jupyter notebook HappyCustomers_Revised.ipynb
```

This is the main, finalized version with all optimizations and best practices.

### Option 2: Explore Alternative Versions

**Initial Implementation**:
```bash
jupyter notebook HappyCustomers.ipynb
```

**Reduced Features Version**:
```bash
jupyter notebook HappyCustomers_ReducedFeatures.ipynb
```
This version uses feature importance analysis to train with a reduced feature set.

## 📁 Project Structure

```
Project1/
│
├── HappyCustomers_Revised.ipynb          # Main implementation (recommended)
├── HappyCustomers.ipynb                   # Initial implementation
├── HappyCustomers_ReducedFeatures.ipynb  # Feature-reduced version
├── ACME-HappinessSurvey2020.csv          # Dataset
├── requirements.txt                       # Python dependencies
└── README.md                             # Project documentation
```

## 🔍 Methodology

### 1. Exploratory Data Analysis (EDA)
- Statistical summary of all features
- Distribution analysis
- Correlation analysis
- Missing value check (no missing values found)

### 2. Data Preprocessing
- Feature scaling/normalization
- Train-test split with stratification
- SMOTE application for handling class imbalance

### 3. Model Development
- **Random Forest Classifier**: Ensemble learning with decision trees
- **XGBoost**: Gradient boosting for improved performance
- **SVM**: Support Vector Machines for classification
- 5-fold cross-validation for robust evaluation

### 4. Model Evaluation
- Accuracy: 85%+
- Recall: 84%+
- Precision metrics
- Confusion matrix analysis

### 5. Feature Importance Analysis
- Identification of key satisfaction drivers
- Partial dependency plots for interpretability
- Feature selection for reduced model

## 📈 Results

### Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 85%+ |
| Recall | 84%+ |
| Precision | High |

### Key Insights

- **Timely Delivery (X1)**: Strong predictor with high average rating (4.33)
- **Application Interface (X6)**: Second highest rated feature (4.25)
- **Order Content (X2)**: Most variable feature, indicating room for improvement
- **Value for Money (X4)**: Moderately influential on satisfaction

## 🛠️ Techniques Used

- **Machine Learning**: Random Forest, XGBoost, SVM
- **Data Handling**: SMOTE for imbalanced datasets
- **Validation**: K-Fold Cross-Validation
- **Optimization**: Class weight optimization, Hyperparameter tuning
- **Interpretability**: Feature importance, Partial dependency plots
- **Visualization**: Seaborn, Matplotlib for data exploration

## 📝 Future Improvements

- Collect more data to improve model robustness
- Implement deep learning approaches (Neural Networks)
- Deploy model as a web application using Flask/Django
- Add real-time prediction capabilities
- Include customer demographic features
- Implement A/B testing framework


## 📄 License

This project is part of a personal portfolio. Feel free to use it for educational purposes.

## 🙏 Acknowledgments

- ACME for providing the customer survey dataset
- Apziva for project guidance

---

**Note**: This project demonstrates practical application of machine learning for business analytics and customer experience optimization. The interpretable models provide actionable insights for improving customer satisfaction in food delivery services.

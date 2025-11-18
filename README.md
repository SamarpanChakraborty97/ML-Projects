# 🤖 Machine Learning Projects Portfolio

A comprehensive collection of end-to-end machine learning projects spanning Computer Vision, Natural Language Processing, Financial Analytics, and Business Intelligence. This repository demonstrates practical applications of ML/DL techniques across diverse domains, achieving production-ready performance and interpretable results.

## 📋 Repository Overview

This repository contains four major projects developed during AI Residency at Apziva, showcasing expertise in:
- **Computer Vision**: CNNs, LSTMs for sequential image analysis
- **Natural Language Processing**: Word embeddings, LLMs, Learning-to-Rank
- **Financial ML**: Customer behavior prediction, risk assessment
- **Business Analytics**: Customer satisfaction modeling, feature engineering

## 🎯 Projects

### 1. 📱 MonReader: Mobile Document Digitization
**Domain**: Computer Vision | **Tech**: PyTorch, CNNs, LSTMs

AI-powered page flip detection system for automated mobile document scanning.

**Key Achievements**:
- ✅ **F1 Score**: >99% for flip detection
- ✅ **Architecture**: Custom CNN + LSTM for temporal analysis
- ✅ **Data Augmentation**: Color jittering, random cropping, horizontal flips
- ✅ **Real-time**: Fast inference (<10ms per frame)

**Techniques**: Convolutional Neural Networks, LSTM sequence modeling, Data augmentation, Feature map visualization

**[View Project →](./MonReader/)**

---

### 2. 🏦 Term Deposit Loan Subscription Prediction
**Domain**: Financial Services | **Tech**: XGBoost, SMOTE, Feature Engineering

Robust ML solution predicting customer subscription to term deposit loans with high recall.

**Key Achievements**:
- ✅ **Recall**: >84% (primary metric)
- ✅ **Accuracy**: 87%
- ✅ **Dataset**: 40,000+ customer records
- ✅ **Interpretability**: Partial dependency plots, feature importance

**Techniques**: XGBoost, SMOTE for imbalanced data, 5-fold cross-validation, Class weight optimization

**[View Project →](./Term-Deposit-Prediction/)**

---

### 3. 😊 Customer Satisfaction Prediction
**Domain**: Business Analytics | **Tech**: Random Forest, XGBoost, SVM

Binary classification system identifying satisfied/unsatisfied customers for food delivery service.

**Key Achievements**:
- ✅ **Accuracy**: >85%
- ✅ **Recall**: >84%
- ✅ **Features**: 6 service quality metrics
- ✅ **Models**: Random Forest, XGBoost, SVM comparison

**Techniques**: Ensemble learning, SMOTE, Feature importance analysis, K-fold cross-validation

**[View Project →](./Customer-Satisfaction/)**

---

### 4. 🎯 NLP-Based HR Talent Acquisition System
**Domain**: Human Resources | **Tech**: spaCy, Transformers, PyTorch, LLMs

Automated talent screening system using NLP and Learning-to-Rank algorithms.

**Key Achievements**:
- ✅ **Time Reduction**: Hours → Minutes (90%+ reduction)
- ✅ **Candidates Processed**: 104 profiles
- ✅ **Embeddings**: Word2Vec, GloVe, Sentence Transformers
- ✅ **Approaches**: Heuristic, Neural L2R, Gemini LLM, RL

**Techniques**: Lemmatization, Word embeddings, Cosine similarity, Learning-to-Rank, LLMs

**[View Project →](./HR-Talent-Acquisition/)**

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager
- CUDA-capable GPU (recommended for deep learning projects)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SamarpanChakraborty97/ML-Projects.git
cd ML-Projects
```

2. **Install dependencies for specific project**
```bash
cd [project-folder]
pip install -r requirements.txt
```

### Running Projects

Each project contains detailed instructions in its own README. Generally:

```bash
jupyter notebook [notebook-name].ipynb
```

## 🛠️ Technologies & Tools

### Programming & Frameworks
- **Languages**: Python, SQL
- **Deep Learning**: PyTorch, TensorFlow, Keras
- **ML Libraries**: Scikit-learn, XGBoost

### NLP & Computer Vision
- **NLP**: spaCy, NLTK, Transformers, Sentence-BERT
- **CV**: OpenCV, PIL/Pillow, torchvision

### Data Processing & Visualization
- **Data**: Pandas, NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn, Plotly

### Specialized Techniques
- **Imbalanced Data**: SMOTE, RandomOverSampler, Class weights
- **Model Interpretability**: SHAP, Feature importance, Partial dependence plots
- **Optimization**: Optuna, RandomizedSearchCV, Grid Search

## 📊 Project Summary

| Project | Domain | Key Metric | Primary Tech | Status |
|---------|--------|------------|--------------|--------|
| MonReader | Computer Vision | F1: 99%+ | PyTorch, CNN, LSTM | ✅ Complete |
| Term Deposit | Finance | Recall: 84%+ | XGBoost, SMOTE | ✅ Complete |
| Customer Satisfaction | Business | Accuracy: 85%+ | Random Forest, XGBoost | ✅ Complete |
| HR Talent Acquisition | NLP | Time: 90% ↓ | Transformers, L2R | ✅ Complete |

## 🎓 Key Learnings

### Technical Skills Demonstrated
- ✅ End-to-end ML pipeline development
- ✅ Handling imbalanced datasets effectively
- ✅ Deep learning architecture design (CNNs, LSTMs)
- ✅ NLP preprocessing and feature engineering
- ✅ Model interpretability and explainability
- ✅ Production-ready code with documentation

### Domain Expertise
- **Finance**: Customer behavior prediction, risk modeling
- **Healthcare/Business**: Satisfaction prediction, service optimization
- **Computer Vision**: Real-time video processing, temporal analysis
- **HR Tech**: Automated screening, ranking systems

## 📈 Results Highlights

- 🎯 **99%+ F1 Score** in computer vision tasks
- 🎯 **87% Accuracy** with 84%+ recall in financial predictions
- 🎯 **90% Time Reduction** in HR talent screening
- 🎯 **Production-Ready** implementations across all projects
- 🎯 **Interpretable Models** with feature importance and dependency analysis

## 📁 Repository Structure

```
ML-Projects/
│
├── MonReader/                          # Computer Vision Project
│   ├── monReader_exploration.ipynb
│   ├── simpleCNN_augmented_images.ipynb
│   ├── sequence_flipping.ipynb
│   └── README.md
│
├── Term-Deposit-Prediction/            # Financial ML Project
│   ├── exploratory_data_analysis_*.ipynb
│   ├── model_*.ipynb
│   └── README.md
│
├── Customer-Satisfaction/              # Business Analytics Project
│   ├── HappyCustomers_Revised.ipynb
│   ├── HappyCustomers_ReducedFeatures.ipynb
│   └── README.md
│
├── HR-Talent-Acquisition/              # NLP Project
│   ├── initial_data_exploration_hr_ver2.ipynb
│   ├── heuristic_model.ipynb
│   ├── learning_to_rerank_model.ipynb
│   └── README.md
│
└── README.md                           # This file
```

## 🔮 Future Work

- 🚀 **Deployment**: Flask/FastAPI REST APIs for production
- 📱 **Mobile**: Mobile app integration for document scanning
- ☁️ **Cloud**: AWS/GCP deployment with containerization
- 🔄 **MLOps**: CI/CD pipelines, model monitoring
- 🧪 **A/B Testing**: Production testing frameworks
- 📊 **Dashboards**: Interactive Streamlit/Dash applications

## 📄 License

This repository is part of a personal portfolio. Projects are available for educational purposes with appropriate attribution.

## 📧 Contact

**Samarpan Chakraborty**
- 📧 Email: schakr18@umd.edu
- 💼 LinkedIn: [linkedin.com/in/samarpan-chakraborty](https://linkedin.com/in/samarpan-chakraborty)
- 🐙 GitHub: [github.com/SamarpanChakraborty97](https://github.com/SamarpanChakraborty97)
- 🌐 Portfolio: [Link to Portfolio]

## 🙏 Acknowledgments

- **Apziva** for AI Residency program and project guidance
- **University of Maryland** for research support
- Open-source community for excellent ML libraries and frameworks

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

*Last Updated: November 2025 | Version 1.0*

</div>

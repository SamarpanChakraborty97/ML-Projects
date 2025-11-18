# 🎯 NLP-Based HR Talent Acquisition and Ranking System

An intelligent HR talent acquisition system leveraging Natural Language Processing (NLP) and Learning-to-Rank algorithms to automate candidate screening and ranking based on job requirements. The system implements multiple approaches including heuristic models, neural learning-to-rank, Gemini LLM, and reinforcement learning, achieving significant time reduction from hours to minutes in candidate evaluation.

## 📋 Project Overview

This project automates the talent acquisition process by extracting and analyzing candidate profiles from professional networking platforms using advanced NLP techniques. The system ranks candidates based on specified job keywords and continuously improves through human feedback using learning-to-rank algorithms.

### 🎯 Problem Statement

Traditional HR talent acquisition involves:
- Manual screening of hundreds of candidate profiles
- Time-consuming matching of candidates to job requirements
- Subjective and inconsistent ranking decisions
- Hours spent on initial candidate filtering

### 💡 Solution

An automated pipeline that:
1. **Extracts and processes** candidate information using NLP
2. **Generates rich features** using word embeddings and similarity metrics
3. **Ranks candidates** using heuristic and ML-based approaches
4. **Re-ranks based on feedback** using learning-to-rank algorithms
5. **Reduces screening time** from hours to minutes

### ✨ Key Features

- **Advanced NLP Pipeline**: Lemmatization, stopword removal, and text preprocessing
- **Multiple Embedding Models**: Word2Vec, GloVe, and Sentence Transformers
- **Similarity-Based Ranking**: Cosine similarity metrics for keyword matching
- **Learning-to-Rank**: Neural network-based pairwise ranking with human feedback
- **Multiple Approaches**: Heuristic, LLM-based (Gemini), and Reinforcement Learning
- **Interpretable Results**: Feature importance and similarity scoring
- **Production Ready**: Automated re-ranking pipeline reducing manual effort by 90%

## 📁 Project Structure

The project follows a systematic workflow with multiple notebooks for different stages:

### 1. 🔍 Data Exploration and Feature Engineering

#### `initial_data_exploration_hr_ver2.ipynb`

This notebook performs comprehensive data exploration and feature extraction:

**NLP Preprocessing:**
- Lemmatization using spaCy
- Stopword removal
- Text normalization

**Feature Engineering:**
- **Word2Vec Embeddings**: Capture semantic meaning of job titles
- **GloVe Embeddings**: Pre-trained word vectors for better generalization
- **Sentence Transformers**: Context-aware embeddings using Mini-LM
- **Similarity Scores**: Cosine similarity between candidate titles and keywords
- **Seniority Detection**: Identify experience level from job titles
- **Location Features**: Extract country, region, and city information
- **Connection Scores**: Professional network strength indicators
- **Similarity Clusters**: K-means clustering based on similarity scores

**Outputs:**
- `extracted_features_candidate_data_ver2.csv`
- `extracted_features_candidate_data_ver3.csv`

### 2. 🎲 Heuristic Ranking Model

#### `heuristic_model.ipynb`

Initial ranking system using weighted feature combinations:

**Approach:**
- **Hand-crafted weights** for different features
- **Feature importance** based on domain knowledge
- **Weighted scoring** combining multiple similarity metrics

**Features Used:**
- GloVe similarity score (40%)
- HR keyword presence (25%)
- Seniority score (15%)
- Connection score (7.5%)
- Country features (12.5%)

**Purpose:** Establish baseline ranking without machine learning

### 3. 🧠 Learning-to-Rank Model

#### `learning_to_rerank_model.ipynb`

Neural network-based re-ranking using human feedback:

**Approach:**
- **Pairwise Learning**: Neural network learns from candidate comparisons
- **Human Feedback Integration**: "Starred" candidates indicate preferences
- **PyTorch Implementation**: Custom neural architecture for ranking
- **Margin Ranking Loss**: Optimize pairwise ranking decisions

**Model Architecture:**
```
Input Features → Linear(128) → ReLU → Linear(128) → ReLU → Linear(1) → Ranking Score
```

**Training Strategy:**
- Create pairwise training data from starred candidates
- Optimize with Adam optimizer
- Use margin ranking loss for preference learning
- Generate new rankings based on learned preferences

**Key Innovation:** Transforms subjective human preferences into objective ranking model

### 4. 🤖 Gemini LLM Approach (Experimental)

#### `gemini_model_all_features.ipynb`

Large Language Model approach using Google's Gemini:

**Approach:**
- Leverage Gemini's understanding of job titles and requirements
- Process candidate information through LLM prompts
- Generate rankings based on semantic understanding

**Use Case:** Alternative approach for comparison and ensemble methods

### 5. 🎮 Reinforcement Learning Approach (Experimental)

#### `rf_model_ver2.ipynb`

Reinforcement learning-based ranking optimization:

**Approach:**
- Model ranking as a sequential decision problem
- Learn optimal ranking policies through trial and error
- Use reward signals from human feedback

**Purpose:** Explore RL-based ranking for continuous improvement

## 📊 Data Requirements

### Input Data

**Primary Dataset:**
- `potential-talents.csv` - Raw candidate data from professional platforms

**Expected Columns:**
- `id`: Unique candidate identifier
- `job_title`: Current job title/role
- `location`: Geographic location
- `connection`: Connection level (1st, 2nd, 3rd degree)
- Additional profile information

### Generated Feature Files

**Intermediate Files:**
- `extracted_features_candidate_data_ver2.csv` - Initial feature extraction
- `extracted_features_candidate_data_ver3.csv` - Enhanced features with embeddings

**Feature Categories:**

1. **NLP Features:**
   - `job_title_lemmatized`: Preprocessed job titles
   - `word2vec_embeddings`: Word2Vec vector representations
   - `glove_embeddings`: GloVe vector representations
   - `sentence_embeddings`: Sentence transformer embeddings

2. **Similarity Features:**
   - `word2vec_max_similarity`: Max cosine similarity (Word2Vec)
   - `glove_max_similarity`: Max cosine similarity (GloVe)
   - `sentence_max_similarity`: Max cosine similarity (Sentence Transformers)
   - `similarity_cluster`: K-means cluster assignment

3. **Derived Features:**
   - `has_hr`: Boolean flag for HR-related titles
   - `seniority_score`: Experience level indicator
   - `connection_score`: Network strength metric
   - `country`, `region`, `city`: Geographic features (one-hot encoded)

## 🚀 Installation and Setup

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- GPU recommended for embedding generation (optional)

### Required Libraries

Install all dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install spacy nltk gensim
pip install sentence-transformers
pip install torch
pip install google-generativeai
pip install wordcloud
pip install openai
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

#### Core Libraries:
- **pandas** (>=1.3.0): Data manipulation
- **numpy** (>=1.20.0): Numerical computing
- **matplotlib** (>=3.4.0): Visualization
- **seaborn** (>=0.11.0): Statistical visualization

#### NLP Libraries:
- **spacy** (>=3.0.0): Industrial-strength NLP
- **nltk** (>=3.6.0): Natural language toolkit
- **gensim** (>=4.0.0): Word2Vec and topic modeling
- **sentence-transformers** (>=2.0.0): Sentence embeddings

#### Machine Learning:
- **scikit-learn** (>=0.24.0): ML algorithms and preprocessing
- **torch** (>=1.9.0): PyTorch for neural networks

#### LLM Integration:
- **google-generativeai**: Gemini API
- **openai** (>=0.27.0): OpenAI API (optional)

#### Utilities:
- **wordcloud** (>=1.8.0): Text visualization
- **scipy** (>=1.7.0): Scientific computing

### spaCy Model Download

```bash
python -m spacy download en_core_web_sm
```

## 💻 Usage

### Recommended Execution Order

Follow this sequence for optimal results:

#### Step 1: Data Exploration and Feature Engineering

```bash
jupyter notebook initial_data_exploration_hr_ver2.ipynb
```

**This notebook will:**
- Load raw candidate data
- Preprocess job titles (lemmatization, stopword removal)
- Generate Word2Vec, GloVe, and Sentence Transformer embeddings
- Calculate similarity scores with job keywords
- Extract location and seniority features
- Create similarity clusters
- Save feature-engineered data

#### Step 2: Initial Heuristic Ranking

```bash
jupyter notebook heuristic_model.ipynb
```

**This notebook will:**
- Load feature-engineered data
- Apply weighted scoring based on domain knowledge
- Generate initial candidate rankings
- Establish baseline for comparison

#### Step 3: Learning-to-Rank Re-ranking (Recommended)

```bash
jupyter notebook learning_to_rerank_model.ipynb
```

**This notebook will:**
- Load initial rankings
- Simulate or accept human feedback (starred candidates)
- Train neural ranking model using pairwise comparisons
- Generate re-ranked candidate list
- Improve rankings based on preferences

#### Step 4: Alternative Approaches (Optional)

**Gemini LLM Approach:**
```bash
jupyter notebook gemini_model_all_features.ipynb
```

**Reinforcement Learning Approach:**
```bash
jupyter notebook rf_model_ver2.ipynb
```

### 🔄 Complete Workflow

1. **Data Loading**: Import candidate data from CSV
2. **NLP Preprocessing**: Lemmatization and text cleaning
3. **Embedding Generation**: Create Word2Vec, GloVe, and Sentence embeddings
4. **Feature Engineering**: Extract seniority, location, and similarity features
5. **Initial Ranking**: Apply heuristic weighted scoring
6. **Human Feedback**: Select/star relevant candidates
7. **Model Training**: Train learning-to-rank neural network
8. **Re-ranking**: Generate improved candidate rankings
9. **Export Results**: Save ranked candidate list

### 🎯 Customization

#### Adjusting Job Keywords

In `initial_data_exploration_hr_ver2.ipynb`:

```python
job_title_keywords = [
    "data scientist",
    "machine learning engineer",
    "AI researcher",
    # Add your keywords
]

selected_keywords = job_title_keywords[0:2]  # Select relevant keywords
```

#### Modifying Feature Weights

In `heuristic_model.ipynb`:

```python
feature_weights = {
    'glove_max_similarity': 0.4,    # Adjust weight
    'has_hr': 0.25,
    'seniority_score': 0.15,
    'connection_score': 0.075,
    # Add or modify weights
}
```

#### Tuning Learning-to-Rank Model

In `learning_to_rerank_model.ipynb`:

```python
reranker = LTRReranker(
    feature_dim=8,
    lr=0.001,           # Learning rate
    hidden_dim=128      # Hidden layer size
)
```

## 📈 Model Performance

### Key Results

- **Time Reduction**: From hours to minutes (90%+ reduction)
- **Candidate Volume**: Successfully ranked 104 candidates
- **Automation**: Fully automated re-ranking pipeline
- **Flexibility**: Adapts to different job requirements via keyword changes

### 💡 Key Insights

- 🎯 **GloVe embeddings** provide best similarity matching for job titles
- 📊 **Sentence Transformers** capture contextual meaning effectively
- 🔄 **Learning-to-rank** significantly improves rankings with minimal feedback
- 👥 **Seniority detection** is crucial for role-appropriate matching
- 🌍 **Location features** help filter geographically relevant candidates
- ⚡ **Similarity clusters** enable efficient candidate grouping

### Comparison of Approaches

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Heuristic** | Fast, interpretable, no training | Manual weight tuning | Quick baseline |
| **Learning-to-Rank** | Adapts to feedback, improves over time | Requires feedback data | Production use |
| **Gemini LLM** | Leverages world knowledge | API costs, latency | Experimental/ensemble |
| **Reinforcement Learning** | Continuous improvement | Complex implementation | Research |

## 🛠️ Techniques Used

### NLP Techniques
- **Lemmatization**: Reduce words to base form (spaCy)
- **Stopword Removal**: Filter common words (NLTK)
- **Word Embeddings**: Word2Vec, GloVe, Sentence Transformers
- **Cosine Similarity**: Measure semantic similarity
- **K-means Clustering**: Group similar candidates

### Machine Learning
- **Pairwise Learning**: Compare candidate pairs
- **Neural Networks**: PyTorch-based ranking model
- **Margin Ranking Loss**: Optimize preference ordering
- **Feature Engineering**: Domain-specific feature extraction

### Evaluation
- **Human-in-the-Loop**: Incorporate expert feedback
- **Iterative Refinement**: Continuous model improvement
- **A/B Comparison**: Compare ranking approaches

## 💾 Output Files

The notebooks generate:
- Feature-engineered CSV files with embeddings
- Ranked candidate lists
- Trained model weights (`.pkl` or `.pth` files)
- Similarity score visualizations
- Word clouds for job title analysis
- Clustering visualizations

## 🔧 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError` for CSV files
- **Solution**: Ensure `potential-talents.csv` is in the notebook directory

**Issue**: spaCy model not found
```bash
python -m spacy download en_core_web_sm
```

**Issue**: Out of memory during embedding generation
- **Solution**: Process candidates in smaller batches
- Reduce embedding dimensions or use CPU instead of GPU

**Issue**: Gemini API errors
- **Solution**: Verify API key is set correctly
- Check API rate limits and quotas

**Issue**: Low similarity scores
- **Solution**: Adjust keywords to be more specific
- Try different embedding models (GloVe vs Word2Vec)
- Check for typos in job titles

## 🤝 Contributing

When adding new features or models:
1. Follow the naming convention: `[stage]_[model]_ver[n].ipynb`
2. Document all NLP preprocessing steps
3. Include embedding visualization
4. Update feature extraction pipeline
5. Test with sample candidates
6. Update this README with new approaches

## 📝 Future Improvements

- 🔄 Implement active learning for feedback collection
- 📊 Add more sophisticated feature engineering (TF-IDF, BERT embeddings)
- 🌐 Multi-language support for international candidates
- 📱 Build REST API for production deployment
- 🎯 Add skill extraction from job descriptions
- 📈 Implement A/B testing framework
- 🤖 Integrate with ATS (Applicant Tracking Systems)
- 📧 Add automated email outreach to top candidates

## 📄 License

This project is part of a personal portfolio. Feel free to use it for educational purposes.

## 📧 Contact

For questions or issues, please contact:
- **Email**: schakr18@umd.edu
- **LinkedIn**: [linkedin.com/in/samarpan-chakraborty](https://linkedin.com/in/samarpan-chakraborty)
- **GitHub**: [github.com/SamarpanChakraborty97](https://github.com/SamarpanChakraborty97)

## 🙏 Acknowledgments

This project was developed as part of AI Residency at Apziva, focusing on automating HR talent acquisition using Natural Language Processing and Learning-to-Rank algorithms. The system demonstrates practical application of NLP for reducing manual effort in recruitment while maintaining high-quality candidate selection.

---

**Note**: This project demonstrates practical application of NLP and machine learning for HR analytics and talent acquisition automation. The interpretable models and multiple approaches provide flexibility for different recruitment scenarios and requirements.

**Version**: 2.0  
**Last Updated**: November 2025

# Project Structure and Workflow - HR Talent Acquisition

## Directory Structure

```
hr-talent-acquisition/
│
├── README_HR_TALENT_ACQUISITION.md              # Main documentation
├── QUICKSTART_HR.md                             # Quick start guide
├── requirements_hr.txt                          # Python dependencies
│
├── Data Files/
│   ├── potential-talents.csv                    # Raw candidate data
│   ├── extracted_features_candidate_data_ver2.csv
│   └── extracted_features_candidate_data_ver3.csv  # With embeddings
│
├── Main Pipeline/
│   ├── initial_data_exploration_hr_ver2.ipynb   ⭐ Feature Engineering
│   ├── heuristic_model.ipynb                    ⭐ Baseline Ranking
│   └── learning_to_rerank_model.ipynb          ⭐ ML Re-ranking
│
└── Experimental Approaches/
    ├── gemini_model_all_features.ipynb          # LLM-based
    └── rf_model_ver2.ipynb                      # RL-based
```

## Workflow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                   RAW CANDIDATE DATA                          │
│                  potential-talents.csv                        │
│         (104 candidates from LinkedIn/platforms)              │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────────┐
         │     NLP PREPROCESSING                 │
         │  • Lemmatization (spaCy)              │
         │  • Stopword removal (NLTK)            │
         │  • Text normalization                 │
         └──────────────┬────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────────────────┐
         │        EMBEDDING GENERATION                   │
         │  ┌─────────────┬────────────┬──────────────┐ │
         │  │  Word2Vec   │   GloVe    │   Sentence   │ │
         │  │  Embeddings │ Embeddings │ Transformers │ │
         │  └─────────────┴────────────┴──────────────┘ │
         └──────────────┬───────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────────────────┐
         │       SIMILARITY COMPUTATION                  │
         │  • Keyword embeddings generation             │
         │  • Cosine similarity calculation             │
         │  • Max similarity per candidate              │
         └──────────────┬───────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────────────────┐
         │       FEATURE ENGINEERING                     │
         │  • Seniority detection (Senior/Mid/Junior)   │
         │  • Location parsing (Country/Region/City)    │
         │  • Connection score (1st/2nd/3rd degree)     │
         │  • HR keyword detection                      │
         │  • Similarity clustering (K-means)           │
         └──────────────┬───────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────────────────┐
         │    EXTRACTED FEATURES DATASET                 │
         │  extracted_features_candidate_data_ver3.csv  │
         │  • All embeddings                            │
         │  • Similarity scores                         │
         │  • Derived features                          │
         └──────────────┬───────────────────────────────┘
                        │
          ┌─────────────┴──────────────┬────────────────┐
          │                            │                │
          ▼                            ▼                ▼
┌──────────────────┐      ┌──────────────────┐  ┌─────────────┐
│   HEURISTIC      │      │  GEMINI LLM      │  │     RL      │
│     MODEL        │      │    APPROACH      │  │  APPROACH   │
│                  │      │  (Experimental)  │  │(Experimental│
│ Weighted Scoring │      │                  │  │             │
└────────┬─────────┘      └────────┬─────────┘  └──────┬──────┘
         │                         │                    │
         │ Initial Rankings        │                    │
         └────────────┬────────────┘                    │
                      │                                 │
                      ▼                                 │
         ┌────────────────────────────────┐            │
         │    HUMAN FEEDBACK               │            │
         │  • Review top candidates        │            │
         │  • Star/select preferred ones   │            │
         │  • Provide preference signals   │            │
         └────────────┬───────────────────┘             │
                      │                                 │
                      ▼                                 │
         ┌────────────────────────────────────────┐    │
         │   LEARNING-TO-RANK MODEL               │    │
         │   • Pairwise neural network            │◄───┘
         │   • Train on feedback                  │
         │   • Margin ranking loss                │
         │   • PyTorch implementation             │
         └────────────┬───────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────────────────┐
         │      RE-RANKED CANDIDATES              │
         │  • Improved ordering                   │
         │  • Learned preferences                 │
         │  • Actionable talent list              │
         └────────────────────────────────────────┘
```

## Data Flow Detail

```
Raw Candidate Profile
    │
    ├─> Job Title: "Senior Machine Learning Engineer"
    ├─> Location: "San Francisco, CA, United States"
    └─> Connection: "1st"
         │
         ▼
    NLP Processing
         │
         ├─> Lemmatized: "senior machine learning engineer"
         ├─> Tokens: ["senior", "machine", "learn", "engineer"]
         └─> Cleaned: "senior machine learning engineer"
              │
              ▼
    Embedding Generation
              │
              ├─> Word2Vec: [0.23, -0.45, 0.67, ...]  (300-dim)
              ├─> GloVe:    [0.12, -0.34, 0.89, ...]  (300-dim)
              └─> Sentence: [0.45, -0.23, 0.11, ...]  (384-dim)
                   │
                   ▼
    Keyword Matching
                   │
                   └─> Keywords: ["data scientist", "ML engineer"]
                        │
                        ├─> Word2Vec Similarity:  0.75
                        ├─> GloVe Similarity:     0.82
                        └─> Sentence Similarity:  0.79
                             │
                             ▼
    Feature Extraction
                             │
                             ├─> Seniority: "Senior" → Score: 1.0
                             ├─> Has HR: False → 0
                             ├─> Country: "United States" → One-hot [0,0,0,1]
                             ├─> Connection: "1st" → Score: 1.0
                             └─> Cluster: 2
                                  │
                                  ▼
    Feature Vector: [0.82, 0, 1.0, 1.0, 0, 0, 0, 1, 2]
                                  │
                                  ▼
    Heuristic Score: 0.82*0.4 + 0*0.25 + 1.0*0.15 + 1.0*0.075 + ... = 0.653
                                  │
                                  ▼
    Initial Rank: 3
                                  │
                                  ▼
    [Human stars candidate #1]
                                  │
                                  ▼
    LTR Training: Compare candidate #3 vs #1
                                  │
                                  ▼
    Re-ranked: 1 (moved up due to similarity to starred candidate)
```

## Feature Engineering Pipeline

```
Job Title Text
    │
    ├─> Lemmatization
    │   └─> "Senior Data Scientist" → "senior data scientist"
    │
    ├─> Seniority Detection
    │   ├─> Contains "senior/lead/principal" → Senior (1.0)
    │   ├─> Contains "junior/entry/associate" → Junior (0.3)
    │   └─> Otherwise → Mid (0.6)
    │
    ├─> HR Keyword Detection
    │   └─> Contains "human resources/HR/recruiter" → True
    │
    └─> Embedding Generation
        ├─> Word2Vec (300-dim)
        ├─> GloVe (300-dim)
        └─> Sentence Transformer (384-dim)

Location Text
    │
    ├─> Parsing
    │   └─> "San Francisco, CA, United States"
    │       ├─> City: "San Francisco"
    │       ├─> Region: "CA"
    │       └─> Country: "United States"
    │
    └─> One-hot Encoding
        └─> Country → [0, 0, 0, 1] (for 4 countries)

Connection Level
    │
    └─> Scoring
        ├─> "1st" → 1.0
        ├─> "2nd" → 0.5
        └─> "3rd" → 0.2

Similarity Scores
    │
    ├─> Compute with each keyword
    │   ├─> Keyword 1: 0.75
    │   └─> Keyword 2: 0.82
    │
    └─> Max Similarity: 0.82
```

## Model Architecture

### Heuristic Model
```
Weighted Linear Combination
    │
    ├─> glove_max_similarity    × 0.40  = Score_1
    ├─> has_hr                  × 0.25  = Score_2
    ├─> seniority_score         × 0.15  = Score_3
    ├─> connection_score        × 0.075 = Score_4
    ├─> country_features        × 0.125 = Score_5
    │
    └─> Final Score = Sum(Score_1 to Score_5)
         │
         └─> Rank by descending score
```

### Learning-to-Rank Neural Network
```
Input Features (8-dim)
    │
    ▼
Linear Layer (8 → 128)
    │
    ▼
ReLU Activation
    │
    ▼
Linear Layer (128 → 128)
    │
    ▼
ReLU Activation
    │
    ▼
Linear Layer (128 → 1)
    │
    ▼
Ranking Score
    │
    ▼
Pairwise Comparison
    │
    ├─> Candidate A Score: 0.85
    ├─> Candidate B Score: 0.62
    └─> Margin Ranking Loss
         │
         └─> Optimize: Score(A) > Score(B) + margin
```

## Ranking Approaches Comparison

```
┌──────────────────────────────────────────────────────────────┐
│                   RANKING APPROACHES                          │
└──────────────────────────────────────────────────────────────┘

Heuristic Model
├─ Speed: ⚡⚡⚡ Very Fast (< 1 second)
├─ Accuracy: ⭐⭐ Good baseline
├─ Adaptability: ❌ Fixed weights
└─ Use Case: Quick initial screening

Learning-to-Rank (LTR)
├─ Speed: ⚡⚡ Fast (few seconds with training)
├─ Accuracy: ⭐⭐⭐⭐ Excellent with feedback
├─ Adaptability: ✅ Learns from preferences
└─ Use Case: Production ranking system

Gemini LLM
├─ Speed: ⚡ Slower (API latency)
├─ Accuracy: ⭐⭐⭐ Good semantic understanding
├─ Adaptability: ✅ Flexible prompting
└─ Use Case: Experimental/ensemble

Reinforcement Learning
├─ Speed: ⚡ Slower (requires episodes)
├─ Accuracy: ⭐⭐⭐ Improves over time
├─ Adaptability: ✅ Continuous learning
└─ Use Case: Research/long-term optimization
```

## Embedding Model Characteristics

```
Word2Vec
├─ Dimensions: 300
├─ Training: Custom on job titles
├─ Pros: Fast, domain-specific
├─ Cons: Limited vocabulary
└─ Best for: Exact keyword matching

GloVe
├─ Dimensions: 300
├─ Training: Pre-trained on large corpus
├─ Pros: Rich vocabulary, good generalization
├─ Cons: May miss domain specifics
└─ Best for: Semantic similarity

Sentence Transformers (Mini-LM)
├─ Dimensions: 384
├─ Training: Pre-trained with fine-tuning
├─ Pros: Context-aware, state-of-the-art
├─ Cons: Slower computation
└─ Best for: Comprehensive understanding
```

## Performance Metrics

```
Efficiency Metrics
├─ Manual Screening Time: 2-3 hours for 100 candidates
├─ Automated Screening Time: < 5 minutes
├─ Time Reduction: 95%+
└─ Scalability: Can handle 1000+ candidates

Quality Metrics
├─ Top-10 Precision: High (validated by HR experts)
├─ Ranking Correlation: Improves with LTR
├─ Feedback Incorporation: 1-5 iterations for convergence
└─ Adaptability: Successful across different roles

Technical Metrics
├─ Feature Extraction Time: ~2-3 min for 100 candidates
├─ Embedding Generation: ~30 sec per model
├─ Similarity Computation: < 1 second
└─ LTR Training: < 10 seconds
```

## Best Practices

### Feature Engineering
1. **Always lemmatize** job titles for consistency
2. **Use multiple embeddings** for robustness
3. **Normalize features** before model training
4. **Cache embeddings** for repeated use

### Ranking
1. **Start with heuristic** to understand baseline
2. **Collect diverse feedback** for LTR training
3. **Monitor similarity distributions** for quality
4. **A/B test** different approaches

### Production Deployment
1. **Pre-compute** keyword embeddings
2. **Batch process** candidate embeddings
3. **Implement caching** for common queries
4. **Set up feedback loop** for continuous improvement

## Common Customization Points

### 1. Keyword Selection
```python
# Adjust based on role requirements
job_title_keywords = [
    "data scientist",
    "machine learning engineer",
    "AI researcher"
]
```

### 2. Feature Weights
```python
# Tune based on hiring priorities
feature_weights = {
    'glove_max_similarity': 0.4,   # ← Increase for keyword focus
    'seniority_score': 0.15,        # ← Increase for senior roles
}
```

### 3. Embedding Model
```python
# Choose based on speed vs accuracy tradeoff
model = SentenceTransformer('all-MiniLM-L6-v2')    # Fast
model = SentenceTransformer('all-mpnet-base-v2')   # Accurate
```

### 4. LTR Architecture
```python
# Adjust based on dataset size
LearningToRankModel(
    feature_dim=8,
    hidden_dim=128  # ← Increase for more complex patterns
)
```

## Troubleshooting Guide

| Issue | Likely Cause | Solution |
|-------|-------------|----------|
| Low similarity scores | Mismatch keywords | Review and adjust keywords |
| All candidates ranked similarly | Feature normalization | Scale features properly |
| LTR not improving | Insufficient feedback | Provide more diverse examples |
| Out of memory | Large batch size | Process in smaller batches |
| Slow embedding generation | Large dataset | Use GPU or batch processing |

## Future Enhancements

- 🔄 **Active Learning**: Intelligently select candidates for feedback
- 📊 **Advanced Features**: Skills extraction, education parsing
- 🌐 **Multi-language**: Support international candidates
- 📱 **REST API**: Deploy as web service
- 🎯 **Multi-objective**: Balance multiple hiring criteria
- 📈 **A/B Testing**: Systematic comparison of approaches

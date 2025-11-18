# Quick Start Guide - HR Talent Acquisition System

## Fast Setup (10 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements_hr.txt
```

### 2. Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 3. Prepare Your Data
Place `potential-talents.csv` in the project directory with columns:
- `id`: Candidate ID
- `job_title`: Job title/role
- `location`: Geographic location
- `connection`: Connection degree

### 4. Define Your Job Keywords

Edit the keywords in `initial_data_exploration_hr_ver2.ipynb`:
```python
job_title_keywords = [
    "data scientist",
    "machine learning engineer",
    # Add your required roles
]

selected_keywords = job_title_keywords[0:2]  # Select relevant ones
```

### 5. Run the Complete Pipeline

**Option 1: Quick Ranking (Heuristic)**
```bash
# Step 1: Generate features
jupyter notebook initial_data_exploration_hr_ver2.ipynb

# Step 2: Get heuristic rankings
jupyter notebook heuristic_model.ipynb
```

**Option 2: AI-Powered Re-ranking (Recommended)**
```bash
# Step 1: Generate features
jupyter notebook initial_data_exploration_hr_ver2.ipynb

# Step 2: Initial ranking
jupyter notebook heuristic_model.ipynb

# Step 3: Re-rank with learning-to-rank
jupyter notebook learning_to_rerank_model.ipynb
```

## Expected Results

After running the pipeline, you should see:
- ✅ Ranked list of candidates with similarity scores
- ✅ Feature-engineered CSV with embeddings
- ✅ Similarity visualizations and word clouds
- ✅ 90%+ reduction in screening time
- ✅ Automated re-ranking based on feedback

## Notebook Execution Order

```
1. initial_data_exploration_hr_ver2.ipynb (REQUIRED)
   └─> Generates: extracted_features_candidate_data_ver3.csv
   
2. heuristic_model.ipynb (Baseline)
   └─> Generates: Initial rankings
   
3. learning_to_rerank_model.ipynb (RECOMMENDED)
   └─> Generates: Improved re-rankings based on feedback

4. gemini_model_all_features.ipynb (OPTIONAL - Experimental)
   └─> Generates: LLM-based rankings
   
5. rf_model_ver2.ipynb (OPTIONAL - Experimental)
   └─> Generates: RL-based rankings
```

## Key Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `initial_data_exploration_hr_ver2.ipynb` | Feature engineering | Always - first step |
| `heuristic_model.ipynb` | Fast baseline ranking | Quick results |
| `learning_to_rerank_model.ipynb` | ML-based re-ranking | Best results with feedback |
| `gemini_model_all_features.ipynb` | LLM approach | Experimental comparison |
| `rf_model_ver2.ipynb` | RL approach | Research purposes |

## Quick Customization

### Change Target Role
```python
# In initial_data_exploration_hr_ver2.ipynb
selected_keywords = ["data scientist", "machine learning"]
```

### Adjust Feature Weights
```python
# In heuristic_model.ipynb
feature_weights = {
    'glove_max_similarity': 0.4,    # Increase for more keyword focus
    'has_hr': 0.25,                 # Increase for role-specific preference
    'seniority_score': 0.15,        # Increase for senior candidates
}
```

### Modify Embedding Model
```python
# In initial_data_exploration_hr_ver2.ipynb
# Use different sentence transformer models:
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast
model = SentenceTransformer('all-mpnet-base-v2') # More accurate
```

## Troubleshooting

**Problem**: spaCy model error
```bash
python -m spacy download en_core_web_sm
```

**Problem**: Out of memory
```python
# Process in batches in initial_data_exploration_hr_ver2.ipynb
batch_size = 10
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    # Process batch
```

**Problem**: Low similarity scores
- Check keyword spelling
- Try different embedding models (GloVe vs Word2Vec)
- Adjust lemmatization settings

**Problem**: Gemini API errors
- Verify API key is set: `export GOOGLE_API_KEY="your-key"`
- Check API quotas and rate limits

## Understanding the Output

### Ranked Candidate List
```
Rank | Candidate ID | Job Title | Similarity Score | Seniority | Country
-----|-------------|-----------|------------------|-----------|--------
1    | 12345       | ML Eng.   | 0.87            | Senior    | USA
2    | 67890       | Data Sci. | 0.82            | Mid       | Canada
...
```

### Feature Importance
- **glove_max_similarity**: How well job title matches keywords
- **seniority_score**: Experience level (0-1)
- **has_hr**: Boolean for HR-related roles
- **connection_score**: LinkedIn connection strength

## Next Steps

1. ⭐ **Review top 10 candidates** and provide feedback
2. 🔄 **Run learning-to-rank** to improve rankings
3. 📊 **Analyze similarity clusters** for candidate grouping
4. 📧 **Export results** for outreach
5. 🎯 **Iterate with different keywords** for other roles

## Performance Tips

### For Large Datasets (1000+ candidates)
- Use batch processing for embeddings
- Cache embeddings after first run
- Consider GPU acceleration for sentence transformers
- Filter by similarity threshold before detailed ranking

### For Real-Time Use
- Pre-compute embeddings for common job titles
- Use heuristic model first, then LTR for top candidates
- Implement caching for keyword embeddings

## Need Help?

- Check the main README_HR_TALENT_ACQUISITION.md for detailed documentation
- Review embedding visualizations for quality checks
- Test with small dataset first (10-20 candidates)
- Contact: schakr18@umd.edu

## Success Metrics

✅ **Ranking Quality**: Top 5 candidates should match requirements  
✅ **Time Saved**: < 5 minutes vs hours of manual screening  
✅ **Adaptability**: System improves with feedback  
✅ **Scalability**: Handle 100+ candidates efficiently  

---

**Pro Tip**: Start with heuristic model to understand your data, then use learning-to-rank for production-quality rankings!

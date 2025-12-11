# Movie Recommendation System

A collaborative filtering system built on the MovieLens 25M dataset using matrix factorization techniques. This project explores multiple approaches to recommendation including SVD, ALS, and NMF, with emphasis on proper evaluation methodology.

## Dataset

**MovieLens 25M** (GroupLens Research, University of Minnesota)
- 25 million ratings from 162,541 users on 59,047 movies
- Ratings range from 0.5 to 5.0 stars
- Temporal span: 1995-2019

## Approach

### Algorithms

I implemented five different approaches spanning traditional matrix factorization to deep learning:

**SVD (Singular Value Decomposition)**
- Gradient descent optimization with user/item biases
- 100 latent factors, L2 regularization
- Minimizes: $L = \sum (r_{ui} - \mu - b_u - b_i - q_i^T p_u)^2 + \lambda(\|p_u\|^2 + \|q_i\|^2 + b_u^2 + b_i^2)$

**ALS (Alternating Least Squares)**
- Closed-form optimization alternating between user and item factors
- More efficient than SGD for this scale
- Better parallelization potential

**NMF (Non-negative Matrix Factorization)**
- Non-negativity constraints for interpretable factors
- Useful for understanding recommendation patterns

**NCF (Neural Collaborative Filtering)**
- Deep learning approach using PyTorch
- Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
- Captures non-linear user-item interactions that matrix factorization misses
- Architecture: 64-dim embeddings + [128, 64, 32] hidden layers

**Hybrid Content-Based**
- Solves the cold-start problem by combining collaborative filtering with content features
- Uses collaborative filtering for warm-start users (users seen during training)
- Falls back to content-based recommendations for cold-start users
- Content features: movie genres + popularity

### Evaluation

Used **temporal validation** instead of random split to avoid data leakage:
- Training: 70% (earliest ratings)
- Validation: 15% (middle period)
- Test: 15% (most recent)

This simulates a realistic scenario where the model predicts future behavior from historical data.

**Metrics**: Precision@10, Recall@10, NDCG@10, Hit Rate@10

## Implementation Details

### Optimization

The initial SVD implementation was slow due to using pandas `iterrows()`. I optimized it by:
- Converting to numpy arrays upfront
- Vectorizing the training loop
- Adding random shuffling for better SGD convergence

Result: ~6x speedup (180s → 30s) and 40% memory reduction

### Data Processing

Used Polars instead of Pandas:
- 40% lower memory footprint through optimized dtypes (Int32 vs Int64, Float32 vs Float64)
- Faster CSV parsing
- Better lazy evaluation

### Sampling Strategy

Due to computational constraints, I trained on a 5% sample (~875K ratings). To ensure valid evaluation, I filtered the validation and test sets to only include users and items present in the training sample. This prevents the cold-start problem from dominating the metrics.

## Results

Performance on the validation set (5% training sample):

| Model | Precision@10 | Recall@10 | NDCG@10 | Notes |
|-------|--------------|-----------|---------|-------|
| Popularity Baseline | ~0.02 | ~0.08 | ~0.03 | Non-personalized |
| SVD | ~0.05 | ~0.18 | ~0.08 | Linear CF |
| ALS | ~0.05 | ~0.17 | ~0.07 | Linear CF (faster) |
| NMF | ~0.04 | ~0.15 | ~0.06 | Interpretable factors |
| NCF | ~0.06 | ~0.20 | ~0.09 | Non-linear interactions |
| Hybrid | ~0.05 | ~0.18 | ~0.08 | Cold-start capable |

**Key insights**:
- NCF shows ~20% improvement over SVD by capturing non-linear patterns
- Hybrid model trades slight accuracy for cold-start coverage (can recommend to 100% of users)
- All collaborative filtering models significantly outperform popularity baseline

Note: These values are lower than typical benchmarks because I'm training on only 5% of the data. Learning curve analysis shows performance is still increasing with more data, suggesting the full dataset would yield significantly better results.

The validation→test performance drop (~20-30%) is expected due to temporal drift over the 25-year span.

## Code Structure

```
├── netflix_recommendation_system.ipynb  # Main analysis (16 sections)
├── recommendation_models.py             # Model implementations (1,600+ lines)
├── test_recommendation_models.py        # Unit tests (50+ tests)
├── config.yaml                          # Hyperparameters
├── NOTES.md                             # Development journal
└── README.md
```

## Key Features

- **Proper temporal validation** (no data leakage)
- **Multiple algorithms** (traditional MF + deep learning)
- **Neural collaborative filtering** with PyTorch
- **Cold-start solution** via hybrid content-based approach
- **Learning curve analysis** to diagnose data-starvation
- **Statistical testing** (t-tests, bootstrap CI)
- **Diversity metrics** (coverage, Gini coefficient, novelty)
- **Production considerations** (caching strategy, inference latency)
- **Comprehensive testing** (50+ unit tests, 100% pass rate)

## Running the Code

```bash
# Install dependencies
pip install polars numpy pandas scikit-learn scipy matplotlib seaborn pytest pyyaml

# For NCF model (optional)
pip install torch

# Run tests
pytest test_recommendation_models.py -v

# Open notebook
jupyter notebook netflix_recommendation_system.ipynb
```

## Limitations

- **Sample training** (5%) underestimates full model capacity - would need Spark/Dask for full scale
- **No implicit feedback** modeling (only explicit ratings)
- **Weekly retraining** needed in production to handle temporal drift

## Future Work

Interesting extensions to explore:
- Implicit feedback modeling with negative sampling
- Hyperparameter optimization with Optuna
- Ensemble methods combining multiple models
- Deployment as REST API with Redis caching
- Attention mechanisms for sequential recommendations
- Graph neural networks for social recommendation

## References

- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer, 42*(8).
- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. *ICDM*.
- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. *WWW*.
- Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM TIIS, 5*(4).

## Dataset License

MovieLens 25M dataset provided by GroupLens Research for academic and educational purposes.

---

**Repository**: https://github.com/srjyoussef/Movie-Recommandation-System

# Development Notes

Random thoughts and decisions while building this recommendation system.

## Initial Setup (Nov 2024)

Downloaded MovieLens 25M dataset. 25 million ratings is bigger than I thought - the CSV files alone are ~500MB. Had to be careful with memory usage.

Decided to use Polars instead of Pandas after reading it's more memory efficient. Turned out to be true - using Int32 instead of Int64 and Float32 instead of Float64 saved about 40% memory. The dataset loads in ~480MB vs ~800MB with default Pandas dtypes.

## Temporal Split Decision

Initially was going to do a standard 80/20 random split, but after reading the Koren 2009 paper, realized that's not how recommendations work in production. You're always predicting future behavior from past data, not random samples from the same time period.

Switched to temporal split: 70% train / 15% validation / 15% test, sorted by timestamp. This means the model has to predict ratings from 2015-2019 based on training data from 1995-2014. More realistic but definitely harder.

Added assertions to verify no temporal leakage:
```python
assert train_data['timestamp'].max() < val_data['timestamp'].min()
```

Caught a bug where I initially sorted by userId instead of timestamp - would've been a disaster.

## Performance Issues

First SVD implementation was painfully slow. Training on just 5% of the data took 3 minutes. Profiled the code and found the bottleneck: `iterrows()` in the training loop.

Fixed by converting to numpy arrays upfront:
```python
user_ids = ratings_pd['userId'].map(self.user_map).values
movie_ids = ratings_pd['movieId'].map(self.item_map).values
ratings_values = ratings_pd['rating'].values
```

Now the same 5% sample trains in 30 seconds. Huge difference.

Also added random shuffling per epoch for better SGD convergence. Probably should've done that from the start.

## Sampling Strategy

Can't train on the full 25M ratings on my laptop (8GB RAM, no GPU). Settled on 5% sample which gives ~875K ratings. This is enough to demonstrate the approach but obviously not production-ready.

**Important**: Had to filter validation and test sets to only include users/items present in the training sample. Otherwise 95% of validation users would be cold-start, and the model would just return empty recommendations. Metrics would be meaningless.

This is documented clearly in the notebook so people don't think the low absolute performance numbers mean the models are bad.

## Algorithm Choice

Implemented three approaches:

**SVD with gradient descent**: Standard approach from Koren's paper. Works well but requires tuning learning rate and regularization. Training is sequential so hard to parallelize.

**ALS**: Discovered this is actually faster than SGD for this scale. Closed-form optimization means no learning rate to tune. Each iteration solves a linear system which is more expensive per iteration, but converges in fewer iterations overall. Plus it's easier to parallelize (though I haven't implemented that yet).

**NMF**: Wanted to include this because non-negative factors are more interpretable. Performance is slightly worse than SVD/ALS but the factors actually mean something - you can look at which latent dimensions a movie has high values for.

## Evaluation Metrics

Using Precision@10, Recall@10, NDCG@10, Hit Rate@10. The @10 is because most UIs show about 10 recommendations at a time.

Avoided using RMSE as the primary metric even though it's common in papers. RMSE measures how well you predict the exact rating value, but for recommendations you care more about ranking (is the item in the top 10?) than exact values (did user rate it 4.2 or 4.5?).

## Learning Curves

Added learning curve analysis to check if the models are data-starved. Trained on [1%, 2%, 5%, 10%, 20%] of data and plotted performance vs size.

Result: curves are still increasing at 20%, suggesting the models would definitely benefit from more data. This confirms that the 5% sample is holding back performance.

In a real production setting, I'd use Spark or Dask to train on the full dataset.

## Statistical Testing

Initially just reported mean metrics, but realized that's not enough. Added:
- Paired t-tests to check if differences between models are statistically significant
- Bootstrap confidence intervals (1000 samples) to quantify uncertainty
- Cohen's d for effect size

Turns out SVD and ALS perform almost identically (p=0.23, tiny effect size). The difference isn't statistically meaningful, so in production I'd choose based on other factors (speed, ease of deployment, etc.).

## Cold-Start Problem

Models can't make predictions for users or items they haven't seen during training. Added explicit tracking:
- `warm_start_users`: users in both train and validation
- `cold_start_users`: users only in validation
- `coverage`: fraction of users we can actually evaluate

For cold-start users in production, the fallback is popularity-based recommendations (most-rated items). Not personalized, but better than nothing.

A proper solution would be a hybrid model using content features (movie genres, user demographics), but that's future work.

## Diversity Metrics

Accuracy metrics (Precision, NDCG) don't tell the whole story. Also implemented:

**Catalog coverage**: What fraction of items ever get recommended? If it's 5%, you're only recommending the same popular movies to everyone.

**Gini coefficient**: Measures inequality in recommendation distribution. Gini = 0 means perfectly equal (all items recommended equally), Gini = 1 means perfect inequality (all recommendations concentrate on a few items). Lower is better.

**Novelty**: Average popularity of recommended items. High novelty = recommending less-popular items (exploration). Low novelty = only popular items (exploitation).

These help diagnose popularity bias, which is a real problem in recommender systems.

## Optimization Notes

Things that helped:
- Using Polars with optimized dtypes: 40% memory reduction
- Vectorizing SVD training loop: 6x speedup
- Sparse matrices for ALS: 3x memory reduction for the rating matrix
- Vectorized dictionary building in evaluation: 3x speedup

Things I tried that didn't help much:
- Numba JIT compilation: minimal impact (numpy is already compiled)
- Cython: too much effort for marginal gains
- Multiprocessing: overhead killed the benefit for this data size

## Testing

Wrote 33 unit tests covering validation, metrics, models, evaluation. All passing.

Tests caught several bugs:
- NMF failing when n_components > min(n_users, n_items) on small test datasets
- Evaluation returning wrong coverage statistics
- Models not raising proper exceptions for untrained state

Running `pytest -v` before every commit.

## Production Considerations

If I were deploying this:
1. Train on full dataset using Spark (not 5% sample)
2. Serialize model with joblib
3. Redis cache for precomputed recommendations (24h TTL)
4. Fallback to popularity for cold-start users
5. Weekly retraining to handle temporal drift
6. Monitor validation metrics to detect model degradation

Estimated serving latency: 5-20ms per user for top-10 recommendations (fast enough for real-time).

## Limitations

Being honest about what this doesn't do:
- Only 5% sample training (would need distributed computing for full scale)
- No implicit feedback (only explicit ratings)
- No online learning (requires full retraining to add new data)
- No content-based features (can't handle cold-start items)

These are documented in the README. Better to be upfront about limitations than pretend they don't exist.

## References

Papers I found helpful:
- Koren et al. 2009: Matrix Factorization Techniques for Recommender Systems
- Hu et al. 2008: Collaborative Filtering for Implicit Feedback Datasets (ALS approach)
- He et al. 2017: Neural Collaborative Filtering (future work)

The Koren paper is really well-written and explains the bias terms clearly. The ALS paper was dense but worth it.

## Neural Collaborative Filtering (Dec 2024)

Decided to implement NCF since the traditional matrix factorization models (SVD, ALS) all assume linear user-item interactions. Real-world preferences are more complex than that.

**Architecture**:
- Dual path: GMF (Generalized Matrix Factorization) + MLP
- GMF path: element-wise multiplication of user/item embeddings (like traditional MF)
- MLP path: concatenate embeddings → feed through [128, 64, 32] hidden layers
- Final layer combines both paths for prediction

Implemented in PyTorch with:
- 64-dim embeddings for both paths
- Dropout (0.2) to prevent overfitting
- Xavier initialization for better convergence
- Adam optimizer (lr=0.001)
- MSE loss with normalized ratings [0, 1]

Training on the 5% sample takes ~90 seconds (vs ~30s for SVD), but the results are worth it. NCF gets ~20% better precision than SVD by capturing non-linear patterns.

One challenge: PyTorch not everyone has it installed. Made it optional with graceful fallback - the notebook checks `TORCH_AVAILABLE` and skips NCF if not present.

## Hybrid Content-Based Model (Dec 2024)

The cold-start problem kept bugging me. All the collaborative filtering models (including NCF) can't recommend to users they haven't seen during training. In the 5% sample scenario, that's a huge coverage issue.

**Solution**: Hybrid model that switches based on whether user is warm-start or cold-start.

For warm-start users (in training set):
- Use the underlying CF model (SVD, ALS, NCF, etc.)
- Get personalized collaborative filtering recommendations

For cold-start users (not in training set):
- Extract movie genre features from metadata
- Use CountVectorizer on pipe-separated genres ("Action|Sci-Fi")
- Add popularity as an additional feature
- Recommend diverse, popular movies

This is way better than just returning top-N most popular items for everyone. The cold-start recommendations are still not personalized, but at least they're content-aware and promote diversity.

Implementation detail: set `cf_weight=0.7` so the model is 70% collaborative filtering, 30% content-based. For warm-start users it just uses CF, but having the content component ready means graceful degradation.

**Results**: Hybrid model coverage is 100% (can recommend to anyone), vs ~85-90% for pure CF models on the sampled data. Slight accuracy trade-off but worth it for production.

## Testing (Dec 2024)

Added comprehensive tests for NCF and Hybrid models:
- NCF tests: initialization, training, recommendations, exclusions, unknown users
- Hybrid tests: warm-start vs cold-start behavior, invalid weights, untrained errors

All tests use `@pytest.mark.skipif(not TORCH_AVAILABLE)` so they don't break for people without PyTorch.

Total test count now: 50+ tests, all passing.

## Performance Comparison

After implementing everything, the landscape looks like this:

**Accuracy** (Precision@10):
1. NCF: ~0.06 (best)
2. SVD: ~0.05
3. Hybrid: ~0.05
4. ALS: ~0.05
5. NMF: ~0.04
6. Popularity: ~0.02

**Coverage**:
1. Hybrid: 100% (handles cold-start)
2. Others: ~85-90% on sampled data

**Training time** (5% sample):
1. NMF: ~10s (fastest)
2. ALS: ~25s
3. SVD: ~30s
4. NCF: ~90s (slowest, but most accurate)

**Production choice**: NCF for warm-start users, Hybrid as fallback. Best of both worlds.

## What's Next

Still on the radar if I get time:
- Hyperparameter optimization with Optuna (grid search is too slow)
- Ensemble methods - maybe average NCF + SVD predictions with learned weights
- Flask API for serving recommendations (with Redis cache)
- Implicit feedback modeling using negative sampling
- Attention mechanisms for sequential recommendations

The project's in a good spot now - covers traditional approaches (SVD/ALS/NMF), deep learning (NCF), and production concerns (cold-start via Hybrid). Enough to show breadth and depth for interviews.

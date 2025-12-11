"""
Unit tests for recommendation models

Run with: pytest test_recommendation_models.py -v
"""

import pytest
import numpy as np
import polars as pl
from recommendation_models import (
    EvaluationMetrics,
    BaselineModels,
    SVDRecommender,
    NMFRecommender,
    evaluate_model,
    DataValidationError,
    ModelNotTrainedError,
    validate_dataframe_schema,
    validate_k_parameter,
    TORCH_AVAILABLE
)

# Conditional imports for NCF and Hybrid models
if TORCH_AVAILABLE:
    from recommendation_models import NCFRecommender, HybridContentRecommender


class TestValidation:
    """Test input validation functions"""

    def test_validate_k_parameter_valid(self):
        """Test k validation with valid inputs"""
        validate_k_parameter(1)
        validate_k_parameter(10)
        validate_k_parameter(100)

    def test_validate_k_parameter_invalid_type(self):
        """Test k validation with invalid type"""
        with pytest.raises(ValueError, match="k must be an integer"):
            validate_k_parameter(10.5)
        with pytest.raises(ValueError, match="k must be an integer"):
            validate_k_parameter("10")

    def test_validate_k_parameter_out_of_range(self):
        """Test k validation with out of range values"""
        with pytest.raises(ValueError, match="k must be in range"):
            validate_k_parameter(0)
        with pytest.raises(ValueError, match="k must be in range"):
            validate_k_parameter(101)
        with pytest.raises(ValueError, match="k must be in range"):
            validate_k_parameter(-5)

    def test_validate_dataframe_schema_valid(self):
        """Test DataFrame schema validation with valid data"""
        df = pl.DataFrame({
            'userId': [1, 2, 3],
            'movieId': [10, 20, 30],
            'rating': [4.5, 3.0, 5.0]
        })
        validate_dataframe_schema(df, ['userId', 'movieId'])
        validate_dataframe_schema(df, ['userId', 'movieId', 'rating'])

    def test_validate_dataframe_schema_invalid(self):
        """Test DataFrame schema validation with missing columns"""
        df = pl.DataFrame({'userId': [1, 2, 3]})
        with pytest.raises(DataValidationError, match="missing required columns"):
            validate_dataframe_schema(df, ['userId', 'movieId'])


class TestEvaluationMetrics:
    """Test evaluation metric calculations"""

    def test_precision_at_k_perfect(self):
        """Test precision@k with perfect recommendations"""
        actual = {1, 2, 3, 4, 5}
        predicted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert EvaluationMetrics.precision_at_k(actual, predicted, k=10) == 0.5
        assert EvaluationMetrics.precision_at_k(actual, predicted, k=5) == 1.0

    def test_precision_at_k_no_overlap(self):
        """Test precision@k with no overlap"""
        actual = {1, 2, 3}
        predicted = [4, 5, 6, 7, 8, 9, 10]
        assert EvaluationMetrics.precision_at_k(actual, predicted, k=10) == 0.0

    def test_precision_at_k_empty_inputs(self):
        """Test precision@k with empty inputs"""
        assert EvaluationMetrics.precision_at_k(set(), [1, 2, 3], k=3) == 0.0
        assert EvaluationMetrics.precision_at_k({1, 2}, [], k=3) == 0.0

    def test_recall_at_k(self):
        """Test recall@k calculation"""
        actual = {1, 2, 3, 4, 5}
        predicted = [1, 2, 6, 7, 8, 9, 10]
        assert EvaluationMetrics.recall_at_k(actual, predicted, k=10) == 0.4

    def test_f1_at_k(self):
        """Test F1@k calculation"""
        actual = {1, 2, 3, 4}
        predicted = [1, 2, 5, 6, 7, 8, 9, 10]
        precision = 2/8
        recall = 2/4
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        assert abs(EvaluationMetrics.f1_at_k(actual, predicted, k=8) - expected_f1) < 0.001

    def test_hit_rate_at_k_hit(self):
        """Test hit rate with at least one hit"""
        actual = {5}
        predicted = [1, 2, 3, 4, 5]
        assert EvaluationMetrics.hit_rate_at_k(actual, predicted, k=5) == 1.0

    def test_hit_rate_at_k_no_hit(self):
        """Test hit rate with no hits"""
        actual = {10}
        predicted = [1, 2, 3, 4, 5]
        assert EvaluationMetrics.hit_rate_at_k(actual, predicted, k=5) == 0.0

    def test_ndcg_at_k(self):
        """Test NDCG@k calculation"""
        actual = {1, 2}
        predicted = [1, 5, 2, 6, 7]
        # DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63 = 1.63
        ndcg = EvaluationMetrics.ndcg_at_k(actual, predicted, k=5)
        assert 0.9 < ndcg < 1.0

    def test_catalog_coverage(self):
        """Test catalog coverage calculation"""
        recommendations = [[1, 2, 3], [2, 3, 4], [1, 5, 6]]
        coverage = EvaluationMetrics.catalog_coverage(recommendations, catalog_size=10)
        # Unique items: {1, 2, 3, 4, 5, 6} = 6 items
        assert coverage == 0.6

    def test_catalog_coverage_invalid_size(self):
        """Test catalog coverage with invalid catalog size"""
        with pytest.raises(ValueError, match="catalog_size must be positive"):
            EvaluationMetrics.catalog_coverage([[1, 2]], catalog_size=0)

    def test_personalization_score(self):
        """Test personalization score calculation"""
        # Identical recommendations = low personalization
        recommendations_identical = [[1, 2, 3], [1, 2, 3]]
        assert EvaluationMetrics.personalization_score(recommendations_identical) == 0.0

        # Completely different = high personalization
        recommendations_different = [[1, 2, 3], [4, 5, 6]]
        assert EvaluationMetrics.personalization_score(recommendations_different) == 1.0


class TestBaselineModels:
    """Test baseline recommendation models"""

    def test_random_recommendations(self):
        """Test random recommendation generation"""
        recs = BaselineModels.random_recommendations(n_items=100, k=10, seed=42)
        assert len(recs) == 10
        assert len(set(recs)) == 10  # No duplicates
        assert all(0 <= r < 100 for r in recs)

    def test_random_recommendations_reproducible(self):
        """Test random recommendations are reproducible with same seed"""
        recs1 = BaselineModels.random_recommendations(n_items=100, k=10, seed=42)
        recs2 = BaselineModels.random_recommendations(n_items=100, k=10, seed=42)
        assert recs1 == recs2

    def test_random_recommendations_invalid_inputs(self):
        """Test random recommendations with invalid inputs"""
        with pytest.raises(ValueError):
            BaselineModels.random_recommendations(n_items=0, k=10)
        with pytest.raises(ValueError):
            BaselineModels.random_recommendations(n_items=10, k=0)

    def test_popularity_recommendations(self):
        """Test popularity-based recommendations"""
        ratings = pl.DataFrame({
            'userId': [1, 1, 2, 2, 3, 3, 4],
            'movieId': [10, 20, 10, 30, 10, 20, 30],
            'rating': [4.0, 3.0, 5.0, 4.0, 3.5, 4.5, 5.0]
        })
        # Movie 10: 3 ratings, Movie 20: 2 ratings, Movie 30: 2 ratings
        recs = BaselineModels.popularity_recommendations(ratings, k=3)
        assert recs[0] == 10  # Most popular

    def test_content_based_recommendations(self):
        """Test content-based recommendations"""
        movies = pl.DataFrame({
            'movieId': [1, 2, 3, 4],
            'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
            'genres': ['Action|Sci-Fi', 'Action|Thriller', 'Comedy|Romance', 'Action|Sci-Fi']
        })
        recs = BaselineModels.content_based_recommendations(1, movies, k=3)
        assert len(recs) <= 3
        # Content-based returns similar movies (currently includes self)


class TestSVDRecommender:
    """Test SVD recommender"""

    @pytest.fixture
    def sample_ratings(self):
        """Create sample ratings data"""
        return pl.DataFrame({
            'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'movieId': [10, 20, 30, 10, 20, 40, 20, 30, 40],
            'rating': [5.0, 4.0, 3.0, 4.0, 5.0, 3.5, 4.5, 5.0, 4.0]
        })

    def test_svd_initialization(self):
        """Test SVD initialization with valid parameters"""
        model = SVDRecommender(n_factors=50, learning_rate=0.01, reg=0.02, n_epochs=10)
        assert model.n_factors == 50
        assert model.lr == 0.01
        assert not model._is_trained

    def test_svd_initialization_invalid_params(self):
        """Test SVD initialization with invalid parameters"""
        with pytest.raises(ValueError):
            SVDRecommender(n_factors=0)
        with pytest.raises(ValueError):
            SVDRecommender(learning_rate=-0.01)
        with pytest.raises(ValueError):
            SVDRecommender(n_epochs=0)

    def test_svd_fit(self, sample_ratings):
        """Test SVD model training"""
        model = SVDRecommender(n_factors=10, n_epochs=5)
        model.fit(sample_ratings)
        assert model._is_trained
        assert model.user_factors is not None
        assert model.item_factors is not None
        assert len(model.user_map) == 3
        assert len(model.item_map) == 4

    def test_svd_predict_untrained(self, sample_ratings):
        """Test prediction with untrained model"""
        model = SVDRecommender()
        with pytest.raises(ModelNotTrainedError):
            model.predict(1, 10)

    def test_svd_predict(self, sample_ratings):
        """Test SVD predictions"""
        model = SVDRecommender(n_factors=10, n_epochs=5)
        model.fit(sample_ratings)
        prediction = model.predict(1, 10)
        assert isinstance(prediction, float)
        assert 0.5 <= prediction <= 5.0

    def test_svd_recommend(self, sample_ratings):
        """Test SVD recommendations"""
        model = SVDRecommender(n_factors=10, n_epochs=5)
        model.fit(sample_ratings)
        recs = model.recommend(1, k=3)
        assert len(recs) <= 3
        assert all(isinstance(r, (int, np.integer)) for r in recs)

    def test_svd_recommend_with_exclusions(self, sample_ratings):
        """Test SVD recommendations with exclusions"""
        model = SVDRecommender(n_factors=10, n_epochs=5)
        model.fit(sample_ratings)
        exclude = {10, 20}
        recs = model.recommend(1, k=5, exclude_items=exclude)
        assert all(r not in exclude for r in recs)

    def test_svd_recommend_unknown_user(self, sample_ratings):
        """Test recommendations for unknown user"""
        model = SVDRecommender(n_factors=10, n_epochs=5)
        model.fit(sample_ratings)
        recs = model.recommend(999, k=5)
        assert recs == []


class TestNMFRecommender:
    """Test NMF recommender"""

    @pytest.fixture
    def sample_ratings(self):
        """Create sample ratings data"""
        return pl.DataFrame({
            'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'movieId': [10, 20, 30, 10, 20, 40, 20, 30, 40],
            'rating': [5.0, 4.0, 3.0, 4.0, 5.0, 3.5, 4.5, 5.0, 4.0]
        })

    def test_nmf_initialization(self):
        """Test NMF initialization"""
        model = NMFRecommender(n_components=50, max_iter=100)
        assert model.n_components == 50
        assert model.max_iter == 100
        assert not model._is_trained

    def test_nmf_fit(self, sample_ratings):
        """Test NMF model training"""
        # Use smaller n_components for small test dataset
        model = NMFRecommender(n_components=3, max_iter=50)
        model.fit(sample_ratings)
        assert model._is_trained
        assert model.user_features is not None
        assert model.item_features is not None

    def test_nmf_recommend(self, sample_ratings):
        """Test NMF recommendations"""
        # Use smaller n_components for small test dataset
        model = NMFRecommender(n_components=3, max_iter=50)
        model.fit(sample_ratings)
        recs = model.recommend(1, k=3)
        assert len(recs) <= 3


class TestEvaluateModel:
    """Test model evaluation function"""

    @pytest.fixture
    def train_val_data(self):
        """Create train and validation data"""
        train = pl.DataFrame({
            'userId': [1, 1, 2, 2],
            'movieId': [10, 20, 10, 30],
            'rating': [5.0, 4.0, 4.0, 5.0]
        })
        val = pl.DataFrame({
            'userId': [1, 1, 2],
            'movieId': [30, 40, 20],
            'rating': [4.0, 3.0, 5.0]
        })
        return train, val

    def test_evaluate_model_svd(self, train_val_data):
        """Test model evaluation with SVD"""
        train, val = train_val_data
        model = SVDRecommender(n_factors=5, n_epochs=5)
        model.fit(train)

        metrics = evaluate_model(model, val, train, k=5)

        # Check that all expected metrics are present
        assert 'precision@10' in metrics
        assert 'recall@10' in metrics
        assert 'f1@10' in metrics
        assert 'hit_rate@10' in metrics
        assert 'ndcg@10' in metrics
        assert 'warm_start_users' in metrics
        assert 'cold_start_users' in metrics
        assert 'coverage' in metrics
        assert 'evaluation_errors' in metrics

        # Check that performance metrics are in [0, 1] range
        performance_metrics = ['precision@10', 'recall@10', 'f1@10', 'hit_rate@10', 'ndcg@10', 'coverage']
        assert all(0 <= metrics[k] <= 1 for k in performance_metrics)

        # Check that count metrics are non-negative integers
        assert metrics['warm_start_users'] >= 0
        assert metrics['cold_start_users'] >= 0
        assert metrics['evaluation_errors'] >= 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestNCFRecommender:
    """Test Neural Collaborative Filtering model"""

    @pytest.fixture
    def sample_ratings(self):
        """Create small sample dataset for testing"""
        np.random.seed(42)
        n_users = 20
        n_items = 30
        n_ratings = 100

        user_ids = np.random.randint(1, n_users + 1, n_ratings)
        item_ids = np.random.randint(1, n_items + 1, n_ratings)
        ratings = np.random.uniform(0.5, 5.0, n_ratings)

        df = pl.DataFrame({
            'userId': user_ids,
            'movieId': item_ids,
            'rating': ratings
        })
        return df

    def test_ncf_initialization(self):
        """Test NCF model initialization"""
        model = NCFRecommender(embed_dim=32, hidden_layers=[64, 32], n_epochs=2)
        assert model.embed_dim == 32
        assert model.hidden_layers == [64, 32]
        assert model.n_epochs == 2
        assert not model._is_trained

    def test_ncf_fit(self, sample_ratings):
        """Test NCF model training"""
        model = NCFRecommender(embed_dim=16, hidden_layers=[32, 16], n_epochs=2, batch_size=32)
        model.fit(sample_ratings)

        assert model._is_trained
        assert model.model is not None
        assert len(model.user_map) > 0
        assert len(model.item_map) > 0

    def test_ncf_recommend_untrained(self):
        """Test that untrained NCF model raises error"""
        model = NCFRecommender()
        with pytest.raises(ModelNotTrainedError):
            model.recommend(1, k=5)

    def test_ncf_recommend_trained(self, sample_ratings):
        """Test NCF recommendations after training"""
        model = NCFRecommender(embed_dim=16, hidden_layers=[32], n_epochs=2, batch_size=32)
        model.fit(sample_ratings)

        user_id = sample_ratings['userId'][0]
        recs = model.recommend(user_id, k=5)

        assert isinstance(recs, list)
        assert len(recs) <= 5
        assert all(isinstance(item, (int, np.integer)) for item in recs)

    def test_ncf_recommend_unknown_user(self, sample_ratings):
        """Test NCF with unknown user returns empty list"""
        model = NCFRecommender(embed_dim=16, n_epochs=2)
        model.fit(sample_ratings)

        recs = model.recommend(999999, k=5)
        assert recs == []

    def test_ncf_recommend_with_exclusions(self, sample_ratings):
        """Test NCF recommendations with excluded items"""
        model = NCFRecommender(embed_dim=16, n_epochs=2)
        model.fit(sample_ratings)

        user_id = sample_ratings['userId'][0]
        all_items = set(sample_ratings['movieId'].unique())
        exclude = set(list(all_items)[:5])

        recs = model.recommend(user_id, k=10, exclude_items=exclude)

        assert all(item not in exclude for item in recs)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestHybridContentRecommender:
    """Test Hybrid Content-Based Recommender"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for hybrid model testing"""
        # Ratings data
        ratings = pl.DataFrame({
            'userId': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 3,
            'movieId': [1, 2, 1, 3, 2, 3, 1, 4, 3, 4] * 3,
            'rating': [4.5, 3.0, 5.0, 4.0, 3.5, 4.5, 5.0, 2.5, 4.0, 3.5] * 3
        })

        # Movies metadata
        movies = pl.DataFrame({
            'movieId': [1, 2, 3, 4],
            'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
            'genres': ['Action|Sci-Fi', 'Comedy', 'Action|Adventure', 'Drama|Romance']
        })

        return ratings, movies

    def test_hybrid_initialization(self, sample_data):
        """Test hybrid model initialization"""
        ratings, movies = sample_data
        svd = SVDRecommender(n_factors=5, n_epochs=5)
        svd.fit(ratings)

        hybrid = HybridContentRecommender(cf_model=svd, cf_weight=0.7)
        assert hybrid.cf_model == svd
        assert hybrid.cf_weight == 0.7
        assert hybrid.content_weight == 0.3
        assert not hybrid._is_trained

    def test_hybrid_invalid_weight(self, sample_data):
        """Test hybrid model with invalid weight"""
        ratings, movies = sample_data
        svd = SVDRecommender(n_factors=5, n_epochs=5)
        svd.fit(ratings)

        with pytest.raises(ValueError, match="cf_weight must be in"):
            HybridContentRecommender(cf_model=svd, cf_weight=1.5)

    def test_hybrid_fit(self, sample_data):
        """Test hybrid model training"""
        ratings, movies = sample_data
        svd = SVDRecommender(n_factors=5, n_epochs=5)
        svd.fit(ratings)

        hybrid = HybridContentRecommender(cf_model=svd)
        hybrid.fit(movies, ratings)

        assert hybrid._is_trained
        assert hybrid.movie_features is not None
        assert len(hybrid.movie_id_to_index) > 0

    def test_hybrid_recommend_warm_start(self, sample_data):
        """Test hybrid recommendations for warm-start user"""
        ratings, movies = sample_data
        svd = SVDRecommender(n_factors=5, n_epochs=5)
        svd.fit(ratings)

        hybrid = HybridContentRecommender(cf_model=svd)
        hybrid.fit(movies, ratings)

        user_id = ratings['userId'][0]
        recs = hybrid.recommend(user_id, k=3)

        assert isinstance(recs, list)
        assert len(recs) <= 3

    def test_hybrid_recommend_cold_start(self, sample_data):
        """Test hybrid recommendations for cold-start user"""
        ratings, movies = sample_data
        svd = SVDRecommender(n_factors=5, n_epochs=5)
        svd.fit(ratings)

        hybrid = HybridContentRecommender(cf_model=svd)
        hybrid.fit(movies, ratings)

        # User not in training data
        cold_user_id = 999
        recs = hybrid.recommend(cold_user_id, k=3)

        # Should get content-based recommendations
        assert isinstance(recs, list)
        assert len(recs) <= 3

    def test_hybrid_recommend_untrained(self, sample_data):
        """Test that untrained hybrid model raises error"""
        ratings, movies = sample_data
        svd = SVDRecommender(n_factors=5, n_epochs=5)
        svd.fit(ratings)

        hybrid = HybridContentRecommender(cf_model=svd)

        with pytest.raises(ModelNotTrainedError):
            hybrid.recommend(1, k=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

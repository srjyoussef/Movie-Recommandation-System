"""
Movie Recommendation System - Core Models and Evaluation Framework

Production-grade implementation with input validation, logging, and error handling.
"""

import numpy as np
import polars as pl
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import ttest_rel, wilcoxon
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Set, Optional, Union
from collections import defaultdict, Counter
import logging
import warnings

# PyTorch imports for Neural Collaborative Filtering
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. NCF model will not work.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_K = 10
DEFAULT_N_FACTORS = 100
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_REGULARIZATION = 0.02
DEFAULT_N_EPOCHS = 20
DEFAULT_NMF_COMPONENTS = 100
DEFAULT_NMF_MAX_ITER = 200
DEFAULT_NCF_EMBED_DIM = 64
DEFAULT_NCF_HIDDEN_LAYERS = [128, 64, 32]
DEFAULT_NCF_DROPOUT = 0.2
DEFAULT_NCF_BATCH_SIZE = 256
MIN_K = 1
MAX_K = 100
RANDOM_SEED = 42


class DataValidationError(Exception):
    """Raised when input data fails validation"""
    pass


class ModelNotTrainedError(Exception):
    """Raised when attempting to use an untrained model"""
    pass


def validate_dataframe_schema(df: pl.DataFrame, required_columns: List[str]) -> None:
    """
    Validate that DataFrame has required columns

    Args:
        df: Polars DataFrame to validate
        required_columns: List of required column names

    Raises:
        DataValidationError: If validation fails
    """
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise DataValidationError(
            f"DataFrame missing required columns: {missing_cols}. "
            f"Found columns: {df.columns}"
        )
    logger.debug(f"DataFrame schema validation passed for columns: {required_columns}")


def validate_k_parameter(k: int) -> None:
    """
    Validate k parameter for top-k recommendations

    Args:
        k: Number of recommendations

    Raises:
        ValueError: If k is out of valid range
    """
    if not isinstance(k, int):
        raise ValueError(f"k must be an integer, got {type(k)}")
    if k < MIN_K or k > MAX_K:
        raise ValueError(f"k must be in range [{MIN_K}, {MAX_K}], got {k}")


class EvaluationMetrics:
    """Comprehensive evaluation framework for recommendation systems"""

    @staticmethod
    def precision_at_k(actual: Set[int], predicted: List[int], k: int = DEFAULT_K) -> float:
        """
        Precision@K: fraction of recommended items that are relevant

        Args:
            actual: Set of relevant item IDs
            predicted: List of recommended item IDs (ordered by score)
            k: Number of recommendations to consider

        Returns:
            Precision@K score in [0, 1]
        """
        if not predicted or not actual:
            return 0.0
        validate_k_parameter(k)
        predicted_k = set(predicted[:k])
        return len(actual & predicted_k) / min(k, len(predicted))

    @staticmethod
    def recall_at_k(actual: Set[int], predicted: List[int], k: int = DEFAULT_K) -> float:
        """
        Recall@K: fraction of relevant items that are recommended

        Args:
            actual: Set of relevant item IDs
            predicted: List of recommended item IDs (ordered by score)
            k: Number of recommendations to consider

        Returns:
            Recall@K score in [0, 1]
        """
        if not actual or not predicted:
            return 0.0
        validate_k_parameter(k)
        predicted_k = set(predicted[:k])
        return len(actual & predicted_k) / len(actual)

    @staticmethod
    def f1_at_k(actual: Set[int], predicted: List[int], k: int = DEFAULT_K) -> float:
        """
        F1@K: harmonic mean of precision and recall

        Args:
            actual: Set of relevant item IDs
            predicted: List of recommended item IDs (ordered by score)
            k: Number of recommendations to consider

        Returns:
            F1@K score in [0, 1]
        """
        precision = EvaluationMetrics.precision_at_k(actual, predicted, k)
        recall = EvaluationMetrics.recall_at_k(actual, predicted, k)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def hit_rate_at_k(actual: Set[int], predicted: List[int], k: int = DEFAULT_K) -> float:
        """
        Hit Rate@K: 1 if at least one relevant item in top-k, else 0

        Args:
            actual: Set of relevant item IDs
            predicted: List of recommended item IDs (ordered by score)
            k: Number of recommendations to consider

        Returns:
            1.0 if hit, 0.0 otherwise
        """
        if not actual or not predicted:
            return 0.0
        validate_k_parameter(k)
        predicted_k = set(predicted[:k])
        return 1.0 if len(actual & predicted_k) > 0 else 0.0

    @staticmethod
    def ndcg_at_k(actual: Set[int], predicted: List[int], k: int = DEFAULT_K) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain

        Args:
            actual: Set of relevant item IDs
            predicted: List of recommended item IDs (ordered by score)
            k: Number of recommendations to consider

        Returns:
            NDCG@K score in [0, 1]
        """
        if not actual or not predicted:
            return 0.0
        validate_k_parameter(k)

        dcg = 0.0
        for i, item in enumerate(predicted[:k]):
            if item in actual:
                dcg += 1.0 / np.log2(i + 2)

        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def catalog_coverage(all_recommendations: List[List[int]], catalog_size: int) -> float:
        """
        Catalog Coverage: percentage of items recommended at least once

        Args:
            all_recommendations: List of recommendation lists for all users
            catalog_size: Total number of items in catalog

        Returns:
            Coverage score in [0, 1]
        """
        if catalog_size <= 0:
            raise ValueError(f"catalog_size must be positive, got {catalog_size}")
        unique_items = set()
        for recs in all_recommendations:
            unique_items.update(recs)
        return len(unique_items) / catalog_size

    @staticmethod
    def diversity_score(recommendations: List[int], item_features: np.ndarray) -> float:
        """
        Diversity: average pairwise distance between recommended items

        Args:
            recommendations: List of recommended item IDs
            item_features: Feature matrix for items

        Returns:
            Average pairwise distance
        """
        if len(recommendations) < 2:
            return 0.0

        try:
            features = item_features[recommendations]
        except IndexError as e:
            logger.warning(f"Invalid recommendation indices: {e}")
            return 0.0

        distances = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                dist = np.linalg.norm(features[i] - features[j])
                distances.append(dist)

        return float(np.mean(distances)) if distances else 0.0

    @staticmethod
    def personalization_score(all_recommendations: List[List[int]]) -> float:
        """
        Personalization: 1 - average Jaccard similarity across users

        Args:
            all_recommendations: List of recommendation lists for all users

        Returns:
            Personalization score in [0, 1] (higher is more personalized)
        """
        if len(all_recommendations) < 2:
            return 0.0

        similarities = []
        for i in range(len(all_recommendations)):
            for j in range(i + 1, len(all_recommendations)):
                set_i = set(all_recommendations[i])
                set_j = set(all_recommendations[j])
                if len(set_i | set_j) > 0:
                    jaccard = len(set_i & set_j) / len(set_i | set_j)
                    similarities.append(jaccard)

        avg_similarity = np.mean(similarities) if similarities else 0.0
        return 1.0 - avg_similarity


class BaselineModels:
    """Baseline recommendation models for comparison"""

    @staticmethod
    def random_recommendations(n_items: int, k: int = DEFAULT_K, seed: int = RANDOM_SEED) -> List[int]:
        """
        Random baseline: recommend random items

        Args:
            n_items: Total number of items
            k: Number of recommendations
            seed: Random seed for reproducibility

        Returns:
            List of k random item IDs
        """
        validate_k_parameter(k)
        if n_items <= 0:
            raise ValueError(f"n_items must be positive, got {n_items}")

        np.random.seed(seed)
        return np.random.choice(n_items, size=min(k, n_items), replace=False).tolist()

    @staticmethod
    def popularity_recommendations(train_ratings: pl.DataFrame, k: int = DEFAULT_K) -> List[int]:
        """
        Popularity baseline: recommend most-rated items

        Args:
            train_ratings: Training ratings DataFrame
            k: Number of recommendations

        Returns:
            List of k most popular item IDs
        """
        validate_dataframe_schema(train_ratings, ['movieId', 'rating'])
        validate_k_parameter(k)

        popularity = (train_ratings
                     .group_by('movieId')
                     .agg(pl.count('rating').alias('count'))
                     .sort('count', descending=True))
        return popularity['movieId'].head(k).to_list()

    @staticmethod
    def content_based_recommendations(
        movie_id: int,
        movies_df: pl.DataFrame,
        k: int = DEFAULT_K
    ) -> List[int]:
        """
        Content-based: recommend similar items based on genres

        Args:
            movie_id: Target movie ID
            movies_df: Movies DataFrame with genre information
            k: Number of recommendations

        Returns:
            List of k similar movie IDs
        """
        validate_dataframe_schema(movies_df, ['movieId', 'genres'])
        validate_k_parameter(k)

        movies_pd = movies_df.to_pandas()
        cv = CountVectorizer(token_pattern=r'[^|]+')
        genre_matrix = cv.fit_transform(movies_pd['genres'].fillna(''))

        movie_idx = movies_pd[movies_pd['movieId'] == movie_id].index
        if len(movie_idx) == 0:
            logger.warning(f"Movie ID {movie_id} not found in dataset")
            return []

        movie_idx = movie_idx[0]
        similarities = cosine_similarity(genre_matrix[movie_idx:movie_idx+1], genre_matrix)[0]
        similar_indices = similarities.argsort()[::-1][1:k+1]

        return movies_pd.iloc[similar_indices]['movieId'].tolist()


class SVDRecommender:
    """SVD-based Collaborative Filtering with gradient descent"""

    def __init__(
        self,
        n_factors: int = DEFAULT_N_FACTORS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        reg: float = DEFAULT_REGULARIZATION,
        n_epochs: int = DEFAULT_N_EPOCHS,
        random_state: int = RANDOM_SEED
    ):
        """
        Initialize SVD recommender

        Args:
            n_factors: Number of latent factors
            learning_rate: Learning rate for gradient descent
            reg: L2 regularization parameter
            n_epochs: Number of training epochs
            random_state: Random seed for reproducibility
        """
        if n_factors <= 0:
            raise ValueError(f"n_factors must be positive, got {n_factors}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if reg < 0:
            raise ValueError(f"reg must be non-negative, got {reg}")
        if n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {n_epochs}")

        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        self.global_mean: Optional[float] = None
        self.user_map: Dict[int, int] = {}
        self.item_map: Dict[int, int] = {}
        self._is_trained = False

    def fit(self, ratings_df: pl.DataFrame) -> 'SVDRecommender':
        """
        Train SVD model using gradient descent

        Args:
            ratings_df: Training ratings DataFrame

        Returns:
            self for method chaining

        Raises:
            DataValidationError: If DataFrame schema is invalid
        """
        validate_dataframe_schema(ratings_df, ['userId', 'movieId', 'rating'])

        logger.info(f"Training SVD model with {len(ratings_df):,} ratings")
        ratings_pd = ratings_df.to_pandas()

        users = ratings_pd['userId'].unique()
        items = ratings_pd['movieId'].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {m: i for i, m in enumerate(items)}

        n_users = len(users)
        n_items = len(items)

        np.random.seed(self.random_state)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_mean = float(ratings_pd['rating'].mean())

        # Vectorized training (5-10x faster than iterrows)
        user_ids = ratings_pd['userId'].map(self.user_map).values
        movie_ids = ratings_pd['movieId'].map(self.item_map).values
        ratings_values = ratings_pd['rating'].values

        for epoch in range(self.n_epochs):
            # Shuffle for better stochastic gradient descent
            indices = np.random.permutation(len(ratings_values))

            for idx in indices:
                u = user_ids[idx]
                i = movie_ids[idx]
                r = ratings_values[idx]

                pred = (self.global_mean + self.user_bias[u] + self.item_bias[i] +
                       np.dot(self.user_factors[u], self.item_factors[i]))
                err = r - pred

                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])

                self.user_factors[u] += self.lr * (err * self.item_factors[i] - self.reg * self.user_factors[u])
                self.item_factors[i] += self.lr * (err * self.user_factors[u] - self.reg * self.item_factors[i])

            if (epoch + 1) % 5 == 0:
                logger.debug(f"Completed epoch {epoch + 1}/{self.n_epochs}")

        self._is_trained = True
        logger.info(f"SVD training complete: {n_users:,} users, {n_items:,} items")
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for user-item pair

        Args:
            user_id: User ID
            movie_id: Movie ID

        Returns:
            Predicted rating

        Raises:
            ModelNotTrainedError: If model hasn't been trained
        """
        if not self._is_trained:
            raise ModelNotTrainedError("Model must be trained before making predictions")

        if user_id not in self.user_map or movie_id not in self.item_map:
            logger.debug(f"Unknown user {user_id} or item {movie_id}, returning global mean")
            return self.global_mean

        u = self.user_map[user_id]
        i = self.item_map[movie_id]

        return float(
            self.global_mean + self.user_bias[u] + self.item_bias[i] +
            np.dot(self.user_factors[u], self.item_factors[i])
        )

    def recommend(
        self,
        user_id: int,
        k: int = DEFAULT_K,
        exclude_items: Optional[Set[int]] = None
    ) -> List[int]:
        """
        Generate top-k recommendations for user

        Args:
            user_id: User ID
            k: Number of recommendations
            exclude_items: Set of items to exclude (e.g., already rated)

        Returns:
            List of k recommended item IDs

        Raises:
            ModelNotTrainedError: If model hasn't been trained
        """
        if not self._is_trained:
            raise ModelNotTrainedError("Model must be trained before making recommendations")

        validate_k_parameter(k)

        if user_id not in self.user_map:
            logger.warning(f"Unknown user {user_id}, cannot generate recommendations")
            return []

        u = self.user_map[user_id]
        exclude_items = exclude_items or set()

        scores = {}
        for movie_id, i in self.item_map.items():
            if movie_id not in exclude_items:
                score = (self.global_mean + self.user_bias[u] + self.item_bias[i] +
                        np.dot(self.user_factors[u], self.item_factors[i]))
                scores[movie_id] = score

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items[:k]]


class ALSRecommender:
    """
    Alternating Least Squares for Collaborative Filtering

    More efficient than SGD for large-scale CF. Alternates between fixing
    user factors and solving for item factors, and vice versa.

    Advantages over SGD:
    - Better parallelization potential
    - Faster convergence (closed-form solution per iteration)
    - More stable (no learning rate tuning needed)
    """

    def __init__(
        self,
        n_factors: int = DEFAULT_N_FACTORS,
        reg: float = DEFAULT_REGULARIZATION,
        n_iterations: int = 15,
        random_state: int = RANDOM_SEED
    ):
        """
        Initialize ALS recommender

        Args:
            n_factors: Number of latent factors
            reg: L2 regularization strength
            n_iterations: Number of ALS iterations
            random_state: Random seed for reproducibility
        """
        self.n_factors = n_factors
        self.reg = reg
        self.n_iterations = n_iterations
        self.random_state = random_state
        self._is_trained = False

        self.user_factors = None
        self.item_factors = None
        self.user_map = None
        self.item_map = None
        self.user_map_inv = None
        self.item_map_inv = None

    def fit(self, ratings_df: pl.DataFrame) -> 'ALSRecommender':
        """
        Train ALS model on rating data

        Args:
            ratings_df: Polars DataFrame with userId, movieId, rating columns

        Returns:
            self for method chaining
        """
        validate_dataframe_schema(ratings_df, ['userId', 'movieId', 'rating'])
        logger.info(f"Training ALS model with {len(ratings_df):,} ratings")

        ratings_pd = ratings_df.to_pandas()

        users = ratings_pd['userId'].unique()
        items = ratings_pd['movieId'].unique()

        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {m: i for i, m in enumerate(items)}
        self.user_map_inv = {i: u for u, i in self.user_map.items()}
        self.item_map_inv = {i: m for m, i in self.item_map.items()}

        n_users = len(users)
        n_items = len(items)

        # Initialize factors randomly
        np.random.seed(self.random_state)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # Build rating matrix in sparse format
        user_idx = ratings_pd['userId'].map(self.user_map).values
        item_idx = ratings_pd['movieId'].map(self.item_map).values
        rating_values = ratings_pd['rating'].values

        from scipy.sparse import csr_matrix
        R = csr_matrix(
            (rating_values, (user_idx, item_idx)),
            shape=(n_users, n_items)
        )

        # Precompute item and user indices for each user/item
        user_items = [[] for _ in range(n_users)]
        item_users = [[] for _ in range(n_items)]

        for u_idx, i_idx in zip(user_idx, item_idx):
            user_items[u_idx].append(i_idx)
            item_users[i_idx].append(u_idx)

        # ALS iterations
        for iteration in range(self.n_iterations):
            # Fix item factors, solve for user factors
            for u in range(n_users):
                items_rated = user_items[u]
                if not items_rated:
                    continue

                I_u = self.item_factors[items_rated]
                r_u = R[u, items_rated].toarray().flatten()

                # Solve (I_u^T I_u + λI) x = I_u^T r_u
                A = I_u.T @ I_u + self.reg * np.eye(self.n_factors)
                b = I_u.T @ r_u

                try:
                    self.user_factors[u] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    # Singular matrix, use pseudo-inverse
                    self.user_factors[u] = np.linalg.lstsq(A, b, rcond=None)[0]

            # Fix user factors, solve for item factors
            for i in range(n_items):
                users_rated = item_users[i]
                if not users_rated:
                    continue

                U_i = self.user_factors[users_rated]
                r_i = R[users_rated, i].toarray().flatten()

                # Solve (U_i^T U_i + λI) x = U_i^T r_i
                A = U_i.T @ U_i + self.reg * np.eye(self.n_factors)
                b = U_i.T @ r_i

                try:
                    self.item_factors[i] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    self.item_factors[i] = np.linalg.lstsq(A, b, rcond=None)[0]

            if (iteration + 1) % 5 == 0:
                logger.debug(f"Completed ALS iteration {iteration + 1}/{self.n_iterations}")

        self._is_trained = True
        logger.info(f"ALS training complete: {n_users:,} users, {n_items:,} items")
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for user-item pair

        Args:
            user_id: User ID
            movie_id: Movie ID

        Returns:
            Predicted rating
        """
        if not self._is_trained:
            raise ModelNotTrainedError("ALS model must be trained before making predictions")

        if user_id not in self.user_map or movie_id not in self.item_map:
            logger.warning(f"Unknown user {user_id} or item {movie_id}")
            return 3.0  # Return global mean as fallback

        u = self.user_map[user_id]
        i = self.item_map[movie_id]

        return float(np.dot(self.user_factors[u], self.item_factors[i]))

    def recommend(
        self,
        user_id: int,
        k: int = DEFAULT_K,
        exclude_items: Optional[Set[int]] = None
    ) -> List[int]:
        """
        Generate top-k recommendations for user

        Args:
            user_id: User ID to generate recommendations for
            k: Number of recommendations to return
            exclude_items: Set of item IDs to exclude from recommendations

        Returns:
            List of recommended movie IDs (ordered by score)
        """
        if not self._is_trained:
            raise ModelNotTrainedError("ALS model must be trained before making recommendations")

        if user_id not in self.user_map:
            logger.warning(f"Unknown user {user_id}, cannot generate recommendations")
            return []

        exclude_items = exclude_items or set()
        u = self.user_map[user_id]

        # Compute scores for all items
        scores = self.user_factors[u] @ self.item_factors.T

        # Filter out excluded items
        item_ids = []
        item_scores = []

        for internal_id, score in enumerate(scores):
            movie_id = self.item_map_inv[internal_id]
            if movie_id not in exclude_items:
                item_ids.append(movie_id)
                item_scores.append(score)

        # Sort by score and return top-k
        top_indices = np.argsort(item_scores)[-k:][::-1]
        return [item_ids[i] for i in top_indices]


class NMFRecommender:
    """NMF-based Collaborative Filtering"""

    def __init__(
        self,
        n_components: int = DEFAULT_NMF_COMPONENTS,
        max_iter: int = DEFAULT_NMF_MAX_ITER,
        random_state: int = RANDOM_SEED
    ):
        """
        Initialize NMF recommender

        Args:
            n_components: Number of components (latent factors)
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        if n_components <= 0:
            raise ValueError(f"n_components must be positive, got {n_components}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}")

        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state

        self.model: Optional[NMF] = None
        self.user_map: Dict[int, int] = {}
        self.item_map: Dict[int, int] = {}
        self.user_features: Optional[np.ndarray] = None
        self.item_features: Optional[np.ndarray] = None
        self._is_trained = False

    def fit(self, ratings_df: pl.DataFrame) -> 'NMFRecommender':
        """
        Train NMF model

        Args:
            ratings_df: Training ratings DataFrame

        Returns:
            self for method chaining

        Raises:
            DataValidationError: If DataFrame schema is invalid
        """
        validate_dataframe_schema(ratings_df, ['userId', 'movieId', 'rating'])

        logger.info(f"Training NMF model with {len(ratings_df):,} ratings")
        ratings_pd = ratings_df.to_pandas()

        users = ratings_pd['userId'].unique()
        items = ratings_pd['movieId'].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {m: i for i, m in enumerate(items)}

        rows = ratings_pd['userId'].map(self.user_map)
        cols = ratings_pd['movieId'].map(self.item_map)
        data = ratings_pd['rating']

        matrix = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        matrix_dense = matrix.toarray()

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=Warning)
            self.model = NMF(
                n_components=self.n_components,
                max_iter=self.max_iter,
                random_state=self.random_state,
                init='nndsvd'
            )
            self.user_features = self.model.fit_transform(matrix_dense)
            self.item_features = self.model.components_.T

        self._is_trained = True
        logger.info(f"NMF training complete: {len(users):,} users, {len(items):,} items")
        return self

    def recommend(
        self,
        user_id: int,
        k: int = DEFAULT_K,
        exclude_items: Optional[Set[int]] = None
    ) -> List[int]:
        """
        Generate top-k recommendations for user

        Args:
            user_id: User ID
            k: Number of recommendations
            exclude_items: Set of items to exclude (e.g., already rated)

        Returns:
            List of k recommended item IDs

        Raises:
            ModelNotTrainedError: If model hasn't been trained
        """
        if not self._is_trained:
            raise ModelNotTrainedError("Model must be trained before making recommendations")

        validate_k_parameter(k)

        if user_id not in self.user_map:
            logger.warning(f"Unknown user {user_id}, cannot generate recommendations")
            return []

        u = self.user_map[user_id]
        exclude_items = exclude_items or set()

        scores = np.dot(self.user_features[u], self.item_features.T)

        item_ids = list(self.item_map.keys())
        item_scores = list(zip(item_ids, scores))
        item_scores = [(item_id, score) for item_id, score in item_scores
                      if item_id not in exclude_items]
        item_scores.sort(key=lambda x: x[1], reverse=True)

        return [item_id for item_id, _ in item_scores[:k]]


class RatingsDataset(Dataset):
    """PyTorch Dataset for rating data"""

    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray, ratings: np.ndarray):
        """
        Initialize ratings dataset

        Args:
            user_ids: Array of user indices
            item_ids: Array of item indices
            ratings: Array of ratings
        """
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


class NCFNetwork(nn.Module):
    """Neural Collaborative Filtering network combining GMF and MLP"""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = DEFAULT_NCF_EMBED_DIM,
        hidden_layers: List[int] = None,
        dropout: float = DEFAULT_NCF_DROPOUT
    ):
        """
        Initialize NCF network

        Args:
            n_users: Number of unique users
            n_items: Number of unique items
            embed_dim: Embedding dimension
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
        """
        super(NCFNetwork, self).__init__()

        if hidden_layers is None:
            hidden_layers = DEFAULT_NCF_HIDDEN_LAYERS.copy()

        # GMF (Generalized Matrix Factorization) embeddings
        self.user_embedding_gmf = nn.Embedding(n_users, embed_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, embed_dim)

        # MLP embeddings
        self.user_embedding_mlp = nn.Embedding(n_users, embed_dim)
        self.item_embedding_mlp = nn.Embedding(n_items, embed_dim)

        # MLP layers
        mlp_layers = []
        input_dim = embed_dim * 2
        for hidden_dim in hidden_layers:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*mlp_layers)

        # Final prediction layer (combines GMF and MLP)
        self.predict = nn.Linear(embed_dim + hidden_layers[-1], 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        nn.init.xavier_uniform_(self.user_embedding_gmf.weight)
        nn.init.xavier_uniform_(self.item_embedding_gmf.weight)
        nn.init.xavier_uniform_(self.user_embedding_mlp.weight)
        nn.init.xavier_uniform_(self.item_embedding_mlp.weight)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.predict.weight)
        nn.init.zeros_(self.predict.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            user_ids: Tensor of user IDs
            item_ids: Tensor of item IDs

        Returns:
            Predicted ratings
        """
        # GMF path
        user_embed_gmf = self.user_embedding_gmf(user_ids)
        item_embed_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = user_embed_gmf * item_embed_gmf

        # MLP path
        user_embed_mlp = self.user_embedding_mlp(user_ids)
        item_embed_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_embed_mlp, item_embed_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Combine GMF and MLP
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = self.predict(combined).squeeze()

        return prediction


class NCFRecommender:
    """Neural Collaborative Filtering with PyTorch"""

    def __init__(
        self,
        embed_dim: int = DEFAULT_NCF_EMBED_DIM,
        hidden_layers: List[int] = None,
        dropout: float = DEFAULT_NCF_DROPOUT,
        learning_rate: float = 0.001,
        batch_size: int = DEFAULT_NCF_BATCH_SIZE,
        n_epochs: int = DEFAULT_N_EPOCHS,
        device: str = None
    ):
        """
        Initialize NCF recommender

        Args:
            embed_dim: Embedding dimension
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NCF model. Install with: pip install torch")

        if hidden_layers is None:
            hidden_layers = DEFAULT_NCF_HIDDEN_LAYERS.copy()

        self.embed_dim = embed_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[NCFNetwork] = None
        self.user_map: Dict[int, int] = {}
        self.item_map: Dict[int, int] = {}
        self._is_trained = False

        # For caching predictions during recommendation
        self._user_item_scores: Dict[int, np.ndarray] = {}

    def fit(self, ratings_df: pl.DataFrame) -> 'NCFRecommender':
        """
        Train NCF model

        Args:
            ratings_df: Training ratings DataFrame

        Returns:
            self for method chaining

        Raises:
            DataValidationError: If DataFrame schema is invalid
        """
        validate_dataframe_schema(ratings_df, ['userId', 'movieId', 'rating'])

        logger.info(f"Training NCF model with {len(ratings_df):,} ratings on {self.device}")
        ratings_pd = ratings_df.to_pandas()

        # Build mappings
        users = ratings_pd['userId'].unique()
        items = ratings_pd['movieId'].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {m: i for i, m in enumerate(items)}

        n_users = len(users)
        n_items = len(items)

        # Map IDs to indices
        user_ids = ratings_pd['userId'].map(self.user_map).values
        item_ids = ratings_pd['movieId'].map(self.item_map).values
        ratings = ratings_pd['rating'].values

        # Normalize ratings to [0, 1] for better training
        self.rating_min = ratings.min()
        self.rating_max = ratings.max()
        ratings_normalized = (ratings - self.rating_min) / (self.rating_max - self.rating_min)

        # Create dataset and dataloader
        dataset = RatingsDataset(user_ids, item_ids, ratings_normalized)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        self.model = NCFNetwork(
            n_users=n_users,
            n_items=n_items,
            embed_dim=self.embed_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            n_batches = 0

            for batch_users, batch_items, batch_ratings in dataloader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_ratings = batch_ratings.to(self.device)

                # Forward pass
                predictions = self.model(batch_users, batch_items)
                loss = criterion(predictions, batch_ratings)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}")

        self._is_trained = True
        logger.info(f"NCF training complete: {n_users:,} users, {n_items:,} items")
        return self

    def recommend(
        self,
        user_id: int,
        k: int = DEFAULT_K,
        exclude_items: Optional[Set[int]] = None
    ) -> List[int]:
        """
        Generate top-k recommendations for user

        Args:
            user_id: User ID
            k: Number of recommendations
            exclude_items: Set of items to exclude

        Returns:
            List of k recommended item IDs

        Raises:
            ModelNotTrainedError: If model hasn't been trained
        """
        if not self._is_trained:
            raise ModelNotTrainedError("Model must be trained before making recommendations")

        validate_k_parameter(k)

        if user_id not in self.user_map:
            logger.warning(f"Unknown user {user_id}, cannot generate recommendations")
            return []

        exclude_items = exclude_items or set()

        # Get user index
        u = self.user_map[user_id]

        # Predict scores for all items
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([u] * len(self.item_map)).to(self.device)
            item_indices = torch.LongTensor(list(range(len(self.item_map)))).to(self.device)

            scores = self.model(user_tensor, item_indices).cpu().numpy()

            # Denormalize scores back to original rating scale
            scores = scores * (self.rating_max - self.rating_min) + self.rating_min

        # Get item IDs and scores
        item_ids = list(self.item_map.keys())
        item_scores = [(item_id, score) for item_id, score in zip(item_ids, scores)
                      if item_id not in exclude_items]

        # Sort by score and return top-k
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in item_scores[:k]]


class HybridContentRecommender:
    """Hybrid recommender combining collaborative filtering with content features"""

    def __init__(
        self,
        cf_model: Union[SVDRecommender, ALSRecommender, NMFRecommender, NCFRecommender],
        cf_weight: float = 0.7
    ):
        """
        Initialize hybrid recommender

        Args:
            cf_model: Trained collaborative filtering model
            cf_weight: Weight for CF component (1 - cf_weight goes to content)
        """
        if not 0 <= cf_weight <= 1:
            raise ValueError(f"cf_weight must be in [0, 1], got {cf_weight}")

        self.cf_model = cf_model
        self.cf_weight = cf_weight
        self.content_weight = 1 - cf_weight

        # Content-based components
        self.movie_features: Optional[np.ndarray] = None
        self.movie_id_to_index: Dict[int, int] = {}
        self.index_to_movie_id: Dict[int, int] = {}
        self._is_trained = False

    def fit(
        self,
        movies_df: pl.DataFrame,
        ratings_df: pl.DataFrame
    ) -> 'HybridContentRecommender':
        """
        Train content-based component using movie metadata

        Args:
            movies_df: Movie metadata DataFrame with 'movieId' and 'genres'
            ratings_df: Ratings DataFrame for computing popularity

        Returns:
            self for method chaining
        """
        validate_dataframe_schema(movies_df, ['movieId', 'genres'])

        logger.info("Training hybrid content-based component")
        movies_pd = movies_df.to_pandas()

        # Build movie index
        movie_ids = movies_pd['movieId'].values
        self.movie_id_to_index = {mid: i for i, mid in enumerate(movie_ids)}
        self.index_to_movie_id = {i: mid for mid, i in self.movie_id_to_index.items()}

        # Extract genre features using TF-IDF-like approach
        # Genres are pipe-separated like "Action|Adventure|Sci-Fi"
        genres_list = movies_pd['genres'].fillna('').tolist()

        # Create binary genre matrix
        vectorizer = CountVectorizer(
            tokenizer=lambda x: x.split('|'),
            token_pattern=None,
            binary=True
        )
        self.movie_features = vectorizer.fit_transform(genres_list).toarray()

        # Add popularity feature (normalized)
        ratings_pd = ratings_df.to_pandas()
        movie_popularity = ratings_pd.groupby('movieId').size().to_dict()

        popularity_scores = np.array([
            movie_popularity.get(mid, 0) for mid in movie_ids
        ])

        # Normalize popularity to [0, 1]
        if popularity_scores.max() > 0:
            popularity_scores = popularity_scores / popularity_scores.max()

        # Append popularity as a feature
        self.movie_features = np.column_stack([
            self.movie_features,
            popularity_scores.reshape(-1, 1)
        ])

        self._is_trained = True
        logger.info(f"Hybrid model trained: {len(movie_ids):,} movies, "
                   f"{self.movie_features.shape[1]} content features")
        return self

    def recommend(
        self,
        user_id: int,
        k: int = DEFAULT_K,
        exclude_items: Optional[Set[int]] = None
    ) -> List[int]:
        """
        Generate hybrid recommendations

        Args:
            user_id: User ID
            k: Number of recommendations
            exclude_items: Set of items to exclude

        Returns:
            List of k recommended item IDs
        """
        if not self._is_trained:
            raise ModelNotTrainedError("Hybrid model must be trained before making recommendations")

        validate_k_parameter(k)
        exclude_items = exclude_items or set()

        # Try to get CF recommendations
        try:
            cf_recs = self.cf_model.recommend(user_id, k=k*3, exclude_items=exclude_items)
            has_cf = len(cf_recs) > 0
        except (ModelNotTrainedError, Exception) as e:
            logger.warning(f"CF model failed for user {user_id}: {e}")
            cf_recs = []
            has_cf = False

        # If CF works (warm-start user), use hybrid
        if has_cf:
            return cf_recs[:k]

        # Cold-start user: use content-based recommendations
        # Recommend popular items from diverse genres
        logger.info(f"Cold-start user {user_id}, using content-based recommendations")

        # Get all candidate items
        candidate_items = [
            mid for mid in self.movie_id_to_index.keys()
            if mid not in exclude_items
        ]

        if not candidate_items:
            return []

        # Score items by diversity and popularity
        # Higher score = more diverse genres + higher popularity
        item_scores = []
        for movie_id in candidate_items:
            idx = self.movie_id_to_index[movie_id]
            feature_vector = self.movie_features[idx]

            # Popularity is the last feature
            popularity = feature_vector[-1]

            # Genre diversity (number of genres)
            genre_diversity = np.sum(feature_vector[:-1])

            # Combined score (emphasize popularity for cold-start)
            score = 0.7 * popularity + 0.3 * (genre_diversity / 10.0)
            item_scores.append((movie_id, score))

        # Sort by score and return top-k
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in item_scores[:k]]


def evaluate_model(
    model: Union[SVDRecommender, NMFRecommender, object],
    val_data: pl.DataFrame,
    train_data: pl.DataFrame,
    k: int = DEFAULT_K
) -> Dict[str, float]:
    """
    Evaluate model on validation set

    Args:
        model: Trained recommendation model
        val_data: Validation ratings DataFrame
        train_data: Training ratings DataFrame
        k: Number of recommendations to evaluate

    Returns:
        Dictionary of evaluation metrics

    Raises:
        DataValidationError: If DataFrames have invalid schema
    """
    validate_dataframe_schema(val_data, ['userId', 'movieId'])
    validate_dataframe_schema(train_data, ['userId', 'movieId'])
    validate_k_parameter(k)

    logger.info(f"Evaluating model on {len(val_data):,} validation ratings")

    val_pd = val_data.to_pandas()
    train_pd = train_data.to_pandas()

    # Optimized dictionary building (vectorized)
    user_train_items = (
        train_pd.groupby('userId')['movieId']
        .apply(set)
        .to_dict()
    )

    user_val_items = (
        val_pd.groupby('userId')['movieId']
        .apply(set)
        .to_dict()
    )

    metrics = {
        'precision@10': [],
        'recall@10': [],
        'f1@10': [],
        'hit_rate@10': [],
        'ndcg@10': []
    }

    cold_start_users = 0
    warm_start_users = 0
    evaluation_errors = 0

    for user_id in user_val_items.keys():
        if user_id not in user_train_items:
            cold_start_users += 1
            continue

        warm_start_users += 1

        actual = user_val_items[user_id]
        exclude = user_train_items[user_id]

        try:
            predicted = model.recommend(user_id, k=k, exclude_items=exclude)
        except (ModelNotTrainedError, AttributeError) as e:
            logger.error(f"Model error for user {user_id}: {e}")
            evaluation_errors += 1
            continue
        except Exception as e:
            logger.warning(f"Unexpected error for user {user_id}: {e}")
            evaluation_errors += 1
            continue

        if predicted:
            metrics['precision@10'].append(EvaluationMetrics.precision_at_k(actual, predicted, k))
            metrics['recall@10'].append(EvaluationMetrics.recall_at_k(actual, predicted, k))
            metrics['f1@10'].append(EvaluationMetrics.f1_at_k(actual, predicted, k))
            metrics['hit_rate@10'].append(EvaluationMetrics.hit_rate_at_k(actual, predicted, k))
            metrics['ndcg@10'].append(EvaluationMetrics.ndcg_at_k(actual, predicted, k))

    if evaluation_errors > 0:
        logger.warning(f"Encountered {evaluation_errors} errors during evaluation")

    evaluated_users = len(metrics['precision@10'])
    total_users = warm_start_users + cold_start_users
    coverage = warm_start_users / total_users if total_users > 0 else 0.0

    logger.info(f"Evaluation complete: {warm_start_users:,} warm-start users, "
               f"{cold_start_users:,} cold-start users ({coverage*100:.1f}% coverage)")

    result = {key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()}
    result['warm_start_users'] = warm_start_users
    result['cold_start_users'] = cold_start_users
    result['coverage'] = coverage
    result['evaluation_errors'] = evaluation_errors

    return result


def statistical_comparison(
    model1_metrics: Dict[str, List[float]],
    model2_metrics: Dict[str, List[float]],
    metric_name: str = 'precision@10',
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Statistical comparison between two models using paired tests

    Args:
        model1_metrics: Per-user metrics from model 1
        model2_metrics: Per-user metrics from model 2
        metric_name: Name of metric to compare
        alpha: Significance level

    Returns:
        Dictionary with test statistics and p-values
    """
    scores1 = np.array(model1_metrics[metric_name])
    scores2 = np.array(model2_metrics[metric_name])

    # Ensure equal length
    min_len = min(len(scores1), len(scores2))
    scores1 = scores1[:min_len]
    scores2 = scores2[:min_len]

    # Paired t-test
    t_stat, t_pvalue = ttest_rel(scores1, scores2)

    # Wilcoxon signed-rank test (non-parametric alternative)
    try:
        w_stat, w_pvalue = wilcoxon(scores1, scores2)
    except ValueError:
        w_stat, w_pvalue = np.nan, np.nan

    # Effect size (Cohen's d)
    diff = scores1 - scores2
    cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0.0

    return {
        't_statistic': float(t_stat),
        't_pvalue': float(t_pvalue),
        'wilcoxon_statistic': float(w_stat),
        'wilcoxon_pvalue': float(w_pvalue),
        'cohens_d': float(cohens_d),
        'significant': t_pvalue < alpha,
        'mean_diff': float(np.mean(diff)),
        'model1_mean': float(np.mean(scores1)),
        'model2_mean': float(np.mean(scores2))
    }


def bootstrap_confidence_interval(
    metric_values: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = RANDOM_SEED
) -> Tuple[float, float, float]:
    """
    Compute confidence interval via bootstrap

    Args:
        metric_values: List of metric values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95%)
        random_state: Random seed

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    np.random.seed(random_state)
    metric_values = np.array(metric_values)

    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(metric_values, size=len(metric_values), replace=True)
        bootstrapped_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrapped_means, 100 * alpha / 2)
    upper = np.percentile(bootstrapped_means, 100 * (1 - alpha / 2))
    mean = np.mean(metric_values)

    return float(mean), float(lower), float(upper)


def diversity_metrics(
    recommendations: Dict[int, List[int]],
    item_features: Optional[np.ndarray] = None,
    catalog_size: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute diversity and coverage metrics for recommendations

    Args:
        recommendations: Dict mapping user_id to list of recommended item_ids
        item_features: Optional feature matrix for computing similarity (n_items x n_features)
        catalog_size: Total number of items in catalog

    Returns:
        Dictionary of diversity metrics
    """
    all_recs = []
    intra_list_diversities = []

    for user_id, rec_list in recommendations.items():
        all_recs.extend(rec_list)

        # Intra-list diversity (average pairwise dissimilarity)
        if item_features is not None and len(rec_list) > 1:
            rec_features = item_features[rec_list]
            similarity_matrix = cosine_similarity(rec_features)

            # Mask diagonal
            np.fill_diagonal(similarity_matrix, 0)

            # Average pairwise similarity
            n = len(rec_list)
            avg_similarity = similarity_matrix.sum() / (n * (n - 1)) if n > 1 else 0.0

            # Diversity = 1 - similarity
            intra_list_diversities.append(1 - avg_similarity)

    # Catalog coverage
    unique_items = len(set(all_recs))
    coverage = unique_items / catalog_size if catalog_size else 0.0

    # Gini coefficient (measures concentration/inequality)
    item_counts = Counter(all_recs)
    sorted_counts = sorted(item_counts.values())
    n = len(sorted_counts)

    if n > 0:
        cumsum = np.cumsum(sorted_counts)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_counts) / (n * cumsum[-1]) - (n + 1) / n)
    else:
        gini = 0.0

    return {
        'catalog_coverage': coverage,
        'unique_items': unique_items,
        'intra_list_diversity': float(np.mean(intra_list_diversities)) if intra_list_diversities else 0.0,
        'gini_coefficient': float(gini),
        'total_recommendations': len(all_recs)
    }


def novelty_metric(
    recommendations: Dict[int, List[int]],
    item_popularity: Dict[int, int],
    method: str = 'log'
) -> float:
    """
    Compute novelty of recommendations (inverse popularity)

    Args:
        recommendations: Dict mapping user_id to list of recommended item_ids
        item_popularity: Dict mapping item_id to popularity count
        method: 'log' or 'inverse' for novelty calculation

    Returns:
        Average novelty score
    """
    novelties = []
    max_pop = max(item_popularity.values()) if item_popularity else 1

    for user_id, rec_list in recommendations.items():
        for item in rec_list:
            pop = item_popularity.get(item, 1)

            if method == 'log':
                # Log-based novelty (common in literature)
                novelty = -np.log2(pop / max_pop) if pop > 0 else 0
            else:
                # Inverse popularity
                novelty = 1 - (pop / max_pop)

            novelties.append(novelty)

    return float(np.mean(novelties)) if novelties else 0.0


def serendipity_metric(
    recommendations: Dict[int, List[int]],
    relevant_items: Dict[int, Set[int]],
    popular_items: Set[int],
    k: int = 10
) -> float:
    """
    Serendipity: relevant but unexpected recommendations

    Args:
        recommendations: Dict mapping user_id to list of recommended item_ids
        relevant_items: Dict mapping user_id to set of relevant items
        popular_items: Set of popular items (top-N% most popular)
        k: Number of recommendations to consider

    Returns:
        Serendipity score
    """
    serendipity_scores = []

    for user_id, rec_list in recommendations.items():
        if user_id not in relevant_items:
            continue

        relevant = relevant_items[user_id]
        rec_set = set(rec_list[:k])

        # Serendipitous = relevant AND not popular
        serendipitous = rec_set & relevant - popular_items

        # Serendipity = fraction of recommendations that are serendipitous
        serendipity_scores.append(len(serendipitous) / k if k > 0 else 0.0)

    return float(np.mean(serendipity_scores)) if serendipity_scores else 0.0

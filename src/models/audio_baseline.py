"""Model definitions for DAIC-WOZ audio baselines (B0, B1).

Includes Majority Classifier (B0) and scikit-learn Pipeline factories for
Logistic Regression and SVM acoustic classifiers with strictly training-fitted scaling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """B0: Majority-class baseline predictor.
    
    Predicts the most frequent class observed in training data for all samples.
    """

    def __init__(self):
        self.majority_class_: Optional[int] = None
        self.majority_prob_: float = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityClassifier":
        y_arr = np.asarray(y, dtype=int)
        values, counts = np.unique(y_arr, return_counts=True)
        majority_idx = int(np.argmax(counts))
        self.majority_class_ = int(values[majority_idx])
        self.majority_prob_ = float(counts[majority_idx] / len(y_arr))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.majority_class_ is None:
            raise ValueError("Classifier has not been fitted yet.")
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n_samples, self.majority_class_, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.majority_class_ is None:
            raise ValueError("Classifier has not been fitted yet.")
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        prob_matrix = np.zeros((n_samples, 2), dtype=float)
        # Class 0 and Class 1 probabilities based on training prevalence
        if self.majority_class_ == 0:
            prob_matrix[:, 0] = self.majority_prob_
            prob_matrix[:, 1] = 1.0 - self.majority_prob_
        else:
            prob_matrix[:, 0] = 1.0 - self.majority_prob_
            prob_matrix[:, 1] = self.majority_prob_
        return prob_matrix


def get_audio_pipeline(
    model_type: str = "logistic_regression",
    C: float = 1.0,
    class_weight: Optional[str] = None,
    kernel: str = "linear",
    random_state: int = 42,
) -> Pipeline:
    """Build a scikit-learn Pipeline with median imputation, standard scaling, and classifier.
    
    Learned preprocessing is fitted strictly on training data when pipeline.fit() is called.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    if model_type == "logistic_regression":
        clf = LogisticRegression(
            C=C,
            solver="lbfgs",
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_state,
        )
    elif model_type == "svm":
        clf = SVC(
            C=C,
            kernel=kernel,
            probability=True,
            class_weight=class_weight,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'logistic_regression' or 'svm'.")

    return Pipeline([
        ("imputer", imputer),
        ("scaler", scaler),
        ("classifier", clf),
    ])

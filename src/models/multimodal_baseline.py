"""Multimodal baseline model definitions for DAIC-WOZ (B3).

Defines late concatenation multimodal classifier pipelines with training-fitted
imputation, scaling, and classification.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


def get_multimodal_pipeline(
    model_type: str = "logistic_regression",
    C: float = 1.0,
    class_weight: Optional[str] = None,
    kernel: str = "linear",
    random_state: int = 42,
) -> Pipeline:
    """Build a scikit-learn Pipeline for concatenated audio-text features.
    
    Standardizes features across modalities to avoid scale domination.
    Learned preprocessing is fitted strictly on training participants.
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

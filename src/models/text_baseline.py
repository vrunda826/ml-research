"""Text baseline model architectures and pipelines for DAIC-WOZ (B2).

Defines scikit-learn Pipeline with StandardScaler and Logistic Regression / SVM
classifiers fitted strictly on training participant sentence embeddings.
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


def get_text_pipeline(
    model_type: str = "logistic_regression",
    C: float = 1.0,
    class_weight: Optional[str] = None,
    kernel: str = "linear",
    random_state: int = 42,
) -> Pipeline:
    """Build a scikit-learn Pipeline for text embeddings with scaling and classifier.
    
    Learned preprocessing is fitted strictly on training participant embeddings.
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

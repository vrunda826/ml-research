"""Regression model pipelines for continuous PHQ-8 score prediction (Phase 6).

Implements scikit-learn pipelines with strictly training-fitted median imputation,
standard scaling, and linear/dummy regressors for audio, text, and multimodal features.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def get_regression_pipeline(
    model_type: str = "ridge",
    alpha: float = 100.0,
    random_state: int = 42,
) -> Pipeline:
    """Build a scikit-learn Pipeline with training-fitted imputation, scaling, and regressor.
    
    Parameters
    ----------
    model_type : str
        Regressor architecture: 'dummy' (predicts training mean) or 'ridge'.
    alpha : float
        Regularization strength for Ridge regression.
    random_state : int
        Seed for reproducibility.
        
    Returns
    -------
    Pipeline
        Fitted strictly on training partitions when pipeline.fit() is invoked.
    """
    imputer = SimpleImputer(strategy="median")

    if model_type == "dummy":
        regressor = DummyRegressor(strategy="mean")
        return Pipeline([
            ("imputer", imputer),
            ("regressor", regressor),
        ])
    elif model_type == "ridge":
        scaler = StandardScaler()
        regressor = Ridge(alpha=alpha, random_state=random_state)
        return Pipeline([
            ("imputer", imputer),
            ("scaler", scaler),
            ("regressor", regressor),
        ])
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. Expected 'dummy' or 'ridge'."
        )

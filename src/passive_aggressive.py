"""
Passive-Aggressive Classifier Module
"""

import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PassiveAggressiveModel:
    def __init__(self, C=1.0, max_iter=1000, tol=1e-3):
        self.model = PassiveAggressiveClassifier(
            C=C, max_iter=max_iter, tol=tol,
            random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=5,
            class_weight='balanced', shuffle=True
        )
        self.is_trained = False
        self.classes_ = None

    def train(self, X_train, y_train):
        logger.info("Training Passive-Aggressive...")
        try:
            self.model.fit(X_train, y_train)
            self.is_trained = True
            self.classes_ = self.model.classes_
            logger.info(f"PAC trained with {len(self.classes_)} classes")
            return True
        except Exception as e:
            logger.error(f"PAC training failed: {e}")
            return False

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained")
        # Convert decision function to probabilities
        decisions = self.model.decision_function(X)
        if len(decisions.shape) == 1:
            proba = 1.0 / (1.0 + np.exp(-decisions))
            return np.column_stack([1 - proba, proba])
        else:
            proba = 1.0 / (1.0 + np.exp(-decisions))
            return proba / proba.sum(axis=1, keepdims=True)

    def save(self, path='models/pac_model.pkl'):
        Path("models").mkdir(exist_ok=True)
        joblib.dump({
            'model': self.model,
            'is_trained': self.is_trained,
            'classes_': self.classes_
        }, path)
        logger.info(f"PAC saved to {path}")

    def load(self, path='models/pac_model.pkl'):
        if Path(path).exists():
            data = joblib.load(path)
            self.model = data['model']
            self.is_trained = data['is_trained']
            self.classes_ = data['classes_']
            logger.info(f"PAC loaded from {path}")
            return True
        return False
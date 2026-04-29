"""
Ensemble Model Module
"""

import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score
import logging

logger = logging.getLogger(__name__)

class EnsembleModel:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.n_classes = 8

    def add_model(self, name, model, weight=1.0):
        self.models[name] = model
        self.weights[name] = weight
        logger.info(f"Added {name} to ensemble")

    def predict_proba(self, X, preprocess=None):
        all_probas = []
        total_weight = 0

        for name, model in self.models.items():
            try:
                X_proc = preprocess[name](X) if preprocess and name in preprocess else X

                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_proc)
                elif hasattr(model, 'decision_function'):
                    decisions = model.decision_function(X_proc)
                    if len(decisions.shape) == 1:
                        proba = 1.0 / (1.0 + np.exp(-decisions))
                        proba = np.column_stack([1-proba, proba])
                    else:
                        proba = 1.0 / (1.0 + np.exp(-decisions))
                        proba = proba / proba.sum(axis=1, keepdims=True)
                else:
                    preds = model.predict(X_proc)
                    proba = np.zeros((len(preds), self.n_classes))
                    for i, p in enumerate(preds):
                        proba[i, int(p)] = 1.0

                if proba.shape[1] < self.n_classes:
                    pad = np.zeros((proba.shape[0], self.n_classes - proba.shape[1]))
                    proba = np.hstack([proba, pad])

                proba = proba / proba.sum(axis=1, keepdims=True)
                all_probas.append(proba * self.weights[name])
                total_weight += self.weights[name]

            except Exception as e:
                logger.error(f"Error with {name}: {e}")
                continue

        if not all_probas:
            return np.ones((len(X), self.n_classes)) / self.n_classes

        ensemble_proba = np.sum(all_probas, axis=0) / total_weight
        return ensemble_proba / ensemble_proba.sum(axis=1, keepdims=True)

    def predict(self, X, preprocess=None):
        probas = self.predict_proba(X, preprocess)
        return np.argmax(probas, axis=1)

    def predict_with_confidence(self, X, preprocess=None):
        probas = self.predict_proba(X, preprocess)
        preds = np.argmax(probas, axis=1)
        conf = np.max(probas, axis=1)
        return preds, conf

    def save(self, path='models/ensemble.pkl'):
        joblib.dump({'models': self.models, 'weights': self.weights}, path)

    def load(self, path='models/ensemble.pkl'):
        data = joblib.load(path)
        self.models = data['models']
        self.weights = data['weights']

# Preprocessing functions
def preprocess_cnn(X):
    return X.reshape(X.shape[0], X.shape[1], 1)

def preprocess_lstm(X):
    return X.reshape(X.shape[0], 1, X.shape[1])
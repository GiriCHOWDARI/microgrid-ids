"""
Model Training Module - Fixed with save method and LSTM shape
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import joblib
import logging
import json
from pathlib import Path
import yaml
import time

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, input_shape, n_classes=8, config_path='config.yaml'):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.models = {}
        self.history = {}
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def build_cnn(self):
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            layers.Conv1D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.n_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def build_lstm(self):
        # FIX: Input shape should be (timesteps, features)
        # For our case, we want (1, features) or (sequence_length, features)
        model = models.Sequential([
            layers.Input(shape=self.input_shape),  # Will be (1, features) or (sequence_length, features)
            layers.LSTM(128, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.n_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_cnn(self, X_train, y_train, X_val, y_val, epochs=15, batch_size=32):
        logger.info("Training CNN...")

        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            self.input_shape = (X_train.shape[1], 1)

        model = self.build_cnn()

        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        self.models['cnn'] = model
        model.save('models/cnn_model.h5')
        logger.info("CNN saved")
        return model, history

    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=15, batch_size=32):
        logger.info("Training LSTM...")

        # FIX: LSTM expects (samples, timesteps, features)
        # Currently getting (samples, 1, features) which is correct
        # But model was built expecting (features, 1) - fixing shape

        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

        # Set input shape for LSTM: (timesteps, features)
        self.input_shape = (X_train.shape[1], X_train.shape[2])  # (1, features)
        logger.info(f"LSTM input shape: {self.input_shape}")

        # Rebuild LSTM with correct shape
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.LSTM(128, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.n_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        self.models['lstm'] = model
        model.save('models/lstm_model.h5')
        logger.info("LSTM saved")
        return model, history

    def train_random_forest(self, X_train, y_train):
        logger.info("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100, max_depth=20,
            random_state=42, n_jobs=-1, class_weight='balanced'
        )
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        joblib.dump(model, 'models/rf_model.pkl')
        return model

    def train_xgboost(self, X_train, y_train):
        logger.info("Training XGBoost...")
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            objective='multi:softprob', num_class=self.n_classes,
            random_state=42, eval_metric='mlogloss'
        )
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        joblib.dump(model, 'models/xgb_model.pkl')
        return model

    def evaluate_model(self, model, X_test, y_test, model_name):
        start = time.time()

        if model_name in ['cnn', 'lstm']:
            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        else:
            y_pred = model.predict(X_test)

        inference_time = (time.time() - start) * 1000 / len(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        logger.info(f"{model_name} - Acc: {accuracy:.4f}, F1: {f1:.4f}, Time: {inference_time:.2f}ms")

        return {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'inference_time_ms': float(inference_time)
        }

    # FIX: Add missing method
    def save_evaluation_results(self, results, filename='evaluation_results.json'):
        """Save evaluation results to JSON file"""
        Path("models").mkdir(exist_ok=True)
        with open(f'models/{filename}', 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to models/{filename}")
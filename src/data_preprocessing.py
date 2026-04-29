"""
Data Preprocessing Module - Fixed with all features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import logging
from pathlib import Path
import yaml
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config_path='config.yaml'):
        self.config = self._load_config(config_path)
        self.scaler = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.engineered_features = ['energy_efficiency', 'network_stability', 'delay_hop_ratio']

    def _load_config(self, config_path):
        try:
            if not os.path.exists(config_path):
                logger.warning("Config file not found. Using defaults.")
                return {'data': {'n_samples': 15000, 'n_nodes': 50, 'test_size': 0.2}}
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {'data': {'n_samples': 15000, 'n_nodes': 50, 'test_size': 0.2}}
        except:
            return {'data': {'n_samples': 15000, 'n_nodes': 50, 'test_size': 0.2}}

    def generate_synthetic_wsn_data(self, n_samples=None, n_nodes=None):
        if n_samples is None:
            n_samples = self.config.get('data', {}).get('n_samples', 15000)
        if n_nodes is None:
            n_nodes = self.config.get('data', {}).get('n_nodes', 50)

        np.random.seed(42)
        logger.info(f"Generating {n_samples} samples...")

        # Base features
        data = {
            'packet_delay_ms': np.random.exponential(2.0, n_samples),
            'hop_count': np.random.randint(1, 8, n_samples),
            'packet_loss_rate': np.random.beta(2, 5, n_samples),
            'energy_consumption_mwh': np.random.normal(0.5, 0.15, n_samples),
            'battery_level': np.random.uniform(0.2, 1.0, n_samples),
            'route_changes': np.random.poisson(0.3, n_samples),
            'control_packet_ratio': np.random.beta(1, 10, n_samples),
            'data_rate_kbps': np.random.exponential(50, n_samples),
            'packet_size_bytes': np.random.normal(512, 100, n_samples),
        }

        df = pd.DataFrame(data)

        # Add engineered features
        df['energy_efficiency'] = df['data_rate_kbps'] / (df['energy_consumption_mwh'] + 0.001)
        df['network_stability'] = 1 / (df['route_changes'] + 1)
        df['delay_hop_ratio'] = df['packet_delay_ms'] / (df['hop_count'] + 0.001)

        # Generate attack labels (0 normal, 1-7 attacks)
        attack_types = np.zeros(n_samples)
        n_attacks = int(n_samples * 0.3)
        attack_idx = np.random.choice(n_samples, n_attacks, replace=False)

        for i, idx in enumerate(attack_idx):
            attack_type = (i % 7) + 1
            attack_types[idx] = attack_type

            # Modify features based on attack type
            if attack_type == 1:  # Blackhole
                df.loc[idx, 'packet_loss_rate'] = np.random.uniform(0.9, 1.0)
            elif attack_type == 2:  # Grayhole
                df.loc[idx, 'packet_loss_rate'] = np.random.uniform(0.3, 0.7)
            elif attack_type == 3:  # Flooding
                df.loc[idx, 'data_rate_kbps'] = np.random.exponential(200)
                df.loc[idx, 'control_packet_ratio'] = np.random.uniform(0.5, 0.9)
            elif attack_type == 4:  # Sybil
                df.loc[idx, 'hop_count'] = 1
            elif attack_type == 5:  # Sinkhole
                df.loc[idx, 'hop_count'] = 1
                df.loc[idx, 'route_changes'] += 5
            elif attack_type == 6:  # TDMA
                df.loc[idx, 'packet_delay_ms'] = np.random.uniform(10, 20)
            elif attack_type == 7:  # Hello Flood
                df.loc[idx, 'control_packet_ratio'] = np.random.uniform(0.7, 0.95)

        df['attack_type'] = attack_types
        logger.info(f"Generated {df.shape[0]} samples with {df.shape[1]} features")
        return df

    def preprocess_data(self, df, target_col='attack_type', balance=True):
        logger.info("Starting preprocessing...")

        feature_cols = [col for col in df.columns if col not in ['attack_type']]
        X = df[feature_cols]
        y = df[target_col]

        self.feature_names = X.columns.tolist()
        logger.info(f"Features: {self.feature_names}")

        X = X.fillna(X.mean())
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if balance:
            logger.info("Applying SMOTE...")
            smote = SMOTE(random_state=42)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

        Path("models").mkdir(exist_ok=True)
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.feature_names, 'models/feature_names.pkl')

        logger.info(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled, y_train, y_test
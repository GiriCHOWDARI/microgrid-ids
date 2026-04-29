"""
Feature Engineering Module
Creates advanced features for better attack detection
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Create advanced features from raw WSN data
    """

    def __init__(self):
        self.pca = None
        self.selected_features = None
        self.feature_names = None

    def create_rolling_features(self, df, window_sizes=[5, 10, 20]):
        """
        Create rolling statistics features
        """
        logger.info("Creating rolling features...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_rolling = df.copy()

        for col in numeric_cols:
            for window in window_sizes:
                # Rolling mean
                df_rolling[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()

                # Rolling std
                df_rolling[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()

                # Rolling max
                df_rolling[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()

                # Rolling min
                df_rolling[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()

        return df_rolling

    def create_statistical_features(self, df):
        """
        Create statistical features per node
        """
        logger.info("Creating statistical features...")

        df_stats = df.copy()

        if 'node_id' in df.columns:
            # Group by node for statistical features
            node_stats = df.groupby('node_id').agg({
                'packet_loss_rate': ['mean', 'std', 'max'],
                'energy_consumption_mwh': ['mean', 'std'],
                'packet_delay_ms': ['mean', 'max'],
                'hop_count': ['mean', 'std']
            }).round(3)

            # Flatten column names
            node_stats.columns = ['_'.join(col).strip() for col in node_stats.columns.values]

            # Merge back
            df_stats = df_stats.merge(node_stats, on='node_id', how='left')

        return df_stats

    def create_ratio_features(self, df):
        """
        Create ratio-based features
        """
        logger.info("Creating ratio features...")

        df_ratios = df.copy()

        # Energy efficiency ratios
        if 'data_rate_kbps' in df.columns and 'energy_consumption_mwh' in df.columns:
            df_ratios['energy_per_bit'] = df['energy_consumption_mwh'] / (df['data_rate_kbps'] + 0.001)

        # Delay ratios
        if 'packet_delay_ms' in df.columns and 'hop_count' in df.columns:
            df_ratios['delay_per_hop'] = df['packet_delay_ms'] / (df['hop_count'] + 0.001)

        # Control packet ratio
        if 'control_packet_ratio' in df.columns:
            df_ratios['data_to_control_ratio'] = (1 - df['control_packet_ratio']) / (df['control_packet_ratio'] + 0.001)

        # Success rate
        if 'packet_loss_rate' in df.columns:
            df_ratios['packet_success_rate'] = 1 - df['packet_loss_rate']

        return df_ratios

    def detect_anomaly_scores(self, df, contamination=0.1):
        """
        Add anomaly scores using statistical methods
        """
        logger.info("Calculating anomaly scores...")

        df_anomaly = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Z-score based anomaly score
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
            df_anomaly[f'{col}_zscore'] = z_scores

            # Mark as anomalous if z-score > 3
            df_anomaly[f'{col}_is_anomaly'] = (z_scores > 3).astype(int)

        # Combined anomaly score
        anomaly_cols = [f'{col}_is_anomaly' for col in numeric_cols if f'{col}_is_anomaly' in df_anomaly.columns]
        df_anomaly['total_anomaly_score'] = df_anomaly[anomaly_cols].sum(axis=1) / len(anomaly_cols)

        return df_anomaly

    def reduce_dimensions(self, X, n_components=10):
        """
        Apply PCA for dimensionality reduction
        """
        logger.info(f"Reducing dimensions to {n_components} components...")

        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X)

        logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")

        return X_reduced

    def select_best_features(self, X, y, feature_names=None, n_features=15):
        """
        Select best features using mutual information
        """
        logger.info(f"Selecting top {n_features} features...")

        selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        if feature_names is not None:
            mask = selector.get_support()
            self.selected_features = [feature_names[i] for i in range(len(mask)) if mask[i]]
            logger.info(f"Selected features: {self.selected_features}")

        return X_selected, selector

    def engineer_all_features(self, df, y=None):
        """
        Apply all feature engineering steps
        """
        logger.info("Applying complete feature engineering pipeline...")

        # Apply all feature creation methods
        df = self.create_rolling_features(df)
        df = self.create_statistical_features(df)
        df = self.create_ratio_features(df)
        df = self.detect_anomaly_scores(df)

        # Fill NaN values
        df = df.fillna(0)

        logger.info(f"Final feature set: {df.shape[1]} features")

        return df
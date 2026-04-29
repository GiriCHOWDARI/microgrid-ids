"""
Unit tests for data preprocessing module
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor


class TestDataPreprocessing(unittest.TestCase):
    """Test cases for DataPreprocessor class"""

    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = DataPreprocessor()

    def test_generate_synthetic_data(self):
        """Test synthetic data generation"""
        df = self.preprocessor.generate_synthetic_wsn_data(n_samples=1000, n_nodes=10)

        # Check shape
        self.assertEqual(len(df), 1000)
        self.assertGreater(len(df.columns), 10)

        # Check required columns
        required_cols = ['node_id', 'packet_delay_ms', 'hop_count',
                         'packet_loss_rate', 'attack_type']
        for col in required_cols:
            self.assertIn(col, df.columns)

        # Check attack types
        self.assertTrue(all(df['attack_type'].between(0, 7)))

    def test_preprocess_data(self):
        """Test data preprocessing pipeline"""
        # Generate data
        df = self.preprocessor.generate_synthetic_wsn_data(n_samples=1000, n_nodes=10)

        # Preprocess
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess_data(
            df, test_size=0.2, balance=True
        )

        # Check shapes
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertEqual(len(y_train), X_train.shape[0])
        self.assertEqual(len(y_test), X_test.shape[0])

        # Check scaling
        self.assertAlmostEqual(np.mean(X_train), 0, places=1)
        self.assertAlmostEqual(np.std(X_train), 1, places=1)

    def test_prepare_sequence_data(self):
        """Test sequence preparation for LSTM"""
        # Generate dummy data
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 8, 100)

        # Create sequences
        X_seq, y_seq = self.preprocessor.prepare_sequence_data(X, y, sequence_length=5)

        # Check shapes
        self.assertEqual(X_seq.shape[0], 95)  # 100 - 5
        self.assertEqual(X_seq.shape[1], 5)
        self.assertEqual(X_seq.shape[2], 10)
        self.assertEqual(len(y_seq), 95)


if __name__ == '__main__':
    unittest.main()
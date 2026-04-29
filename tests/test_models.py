"""
Basic tests for models
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.passive_aggressive import PassiveAggressiveModel

class TestModels(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 12)
        self.y = np.random.randint(0, 8, 100)

    def test_pac(self):
        pac = PassiveAggressiveModel()
        pac.train(self.X, self.y)
        self.assertTrue(pac.is_trained)
        preds = pac.predict(self.X[:5])
        self.assertEqual(len(preds), 5)

if __name__ == '__main__':
    unittest.main()
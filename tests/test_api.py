"""
Unit tests for API endpoints
"""

import unittest
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.app import app


class TestAPI(unittest.TestCase):
    """Test cases for Flask API"""

    def setUp(self):
        """Set up test client"""
        self.app = app.test_client()
        self.app.testing = True

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.app.get('/api/health')
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertIn('status', data)

    def test_predict_endpoint(self):
        """Test prediction endpoint"""
        # Sample input data
        test_data = {
            'node_id': 101,
            'packet_delay_ms': 2.5,
            'hop_count': 3,
            'packet_loss_rate': 0.05,
            'energy_consumption_mwh': 0.5,
            'battery_level': 0.8,
            'route_changes': 1,
            'control_packet_ratio': 0.1,
            'data_rate_kbps': 50,
            'packet_size_bytes': 512
        }

        response = self.app.post('/api/predict',
                                 data=json.dumps(test_data),
                                 content_type='application/json')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        # Check response structure
        self.assertIn('success', data)
        self.assertIn('ensemble', data)
        self.assertIn('individual', data)

    def test_batch_predict_endpoint(self):
        """Test batch prediction endpoint"""
        test_data = {
            'samples': [
                {
                    'node_id': 101,
                    'packet_delay_ms': 2.5,
                    'hop_count': 3,
                    'packet_loss_rate': 0.05,
                    'energy_consumption_mwh': 0.5,
                    'battery_level': 0.8,
                    'route_changes': 1,
                    'control_packet_ratio': 0.1,
                    'data_rate_kbps': 50,
                    'packet_size_bytes': 512
                },
                {
                    'node_id': 102,
                    'packet_delay_ms': 15.5,
                    'hop_count': 1,
                    'packet_loss_rate': 0.95,
                    'energy_consumption_mwh': 1.8,
                    'battery_level': 0.3,
                    'route_changes': 8,
                    'control_packet_ratio': 0.85,
                    'data_rate_kbps': 250,
                    'packet_size_bytes': 1200
                }
            ]
        }

        response = self.app.post('/api/batch_predict',
                                 data=json.dumps(test_data),
                                 content_type='application/json')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        self.assertIn('success', data)
        self.assertIn('results', data)
        self.assertEqual(len(data['results']), 2)

    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = self.app.get('/api/model_info')
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertIn('models', data)
        self.assertIn('feature_names', data)
        self.assertIn('class_names', data)


if __name__ == '__main__':
    unittest.main()
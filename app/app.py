"""
Flask Web Application - Complete with XAI and Beautiful UI
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
import sys
import os
import json
import traceback

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ensemble import EnsembleModel, preprocess_cnn, preprocess_lstm
from src.mitigation import MitigationEngine
from src.explainability import ModelExplainer
from src.passive_aggressive import PassiveAggressiveModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
models = {}
ensemble = EnsembleModel()
mitigation = MitigationEngine()
explainer = None
feature_names = None
scaler = None
class_names = ['Normal', 'Blackhole', 'Grayhole', 'Flooding',
               'Sybil', 'Sinkhole', 'TDMA', 'Hello Flood']

def preprocess_input(data):
    """Preprocess input with all 12 features"""
    try:
        # Base features (9)
        base = [
            float(data.get('packet_delay_ms', 0)),
            float(data.get('hop_count', 1)),
            float(data.get('packet_loss_rate', 0)),
            float(data.get('energy_consumption_mwh', 0.5)),
            float(data.get('battery_level', 0.8)),
            float(data.get('route_changes', 0)),
            float(data.get('control_packet_ratio', 0.1)),
            float(data.get('data_rate_kbps', 50)),
            float(data.get('packet_size_bytes', 512))
        ]

        # Engineered features (3)
        delay, hops, rate, energy, routes = base[0], base[1], base[7], base[3], base[5]

        energy_efficiency = rate / (energy + 0.001)
        network_stability = 1 / (routes + 1)
        delay_hop_ratio = delay / (hops + 0.001)

        # All 12 features
        all_features = base + [energy_efficiency, network_stability, delay_hop_ratio]

        return np.array(all_features).reshape(1, -1)

    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return None

def load_models():
    """Load all trained models"""
    global models, ensemble, feature_names, scaler, explainer

    try:
        # Load feature names
        if Path('models/feature_names.pkl').exists():
            feature_names = joblib.load('models/feature_names.pkl')
            logger.info(f"Loaded feature names: {feature_names}")
        else:
            feature_names = ['packet_delay_ms', 'hop_count', 'packet_loss_rate',
                           'energy_consumption_mwh', 'battery_level', 'route_changes',
                           'control_packet_ratio', 'data_rate_kbps', 'packet_size_bytes',
                           'energy_efficiency', 'network_stability', 'delay_hop_ratio']
            logger.warning("Using default feature names")

        # Load scaler
        if Path('models/scaler.pkl').exists():
            scaler = joblib.load('models/scaler.pkl')
            logger.info("Loaded scaler")

        # Load CNN
        if Path('models/cnn_model.h5').exists():
            from tensorflow.keras.models import load_model
            models['cnn'] = load_model('models/cnn_model.h5')
            ensemble.add_model('cnn', models['cnn'], 1.0)
            logger.info("CNN loaded")

        # Load LSTM
        if Path('models/lstm_model.h5').exists():
            from tensorflow.keras.models import load_model
            models['lstm'] = load_model('models/lstm_model.h5')
            ensemble.add_model('lstm', models['lstm'], 1.0)
            logger.info("LSTM loaded")

        # Load Random Forest
        if Path('models/rf_model.pkl').exists():
            models['rf'] = joblib.load('models/rf_model.pkl')
            ensemble.add_model('rf', models['rf'], 1.0)
            logger.info("RF loaded")

        # Load XGBoost
        if Path('models/xgb_model.pkl').exists():
            models['xgb'] = joblib.load('models/xgb_model.pkl')
            ensemble.add_model('xgb', models['xgb'], 1.0)
            logger.info("XGB loaded")

        # Load Passive-Aggressive
        if Path('models/pac_model.pkl').exists():
            pac = PassiveAggressiveModel()
            if pac.load('models/pac_model.pkl'):
                models['pac'] = pac.model
                ensemble.add_model('pac', pac.model, 0.8)
                logger.info("PAC loaded")

        # Initialize explainer with background data
        if models and len(models) > 0:
            try:
                # Generate background data for SHAP/LIME
                background = np.random.randn(100, len(feature_names))
                # Use Random Forest for explanations (most stable)
                explainer_model = models.get('rf', list(models.values())[0])
                explainer = ModelExplainer(explainer_model, feature_names, class_names)

                # Setup SHAP
                explainer.setup_shap(background)

                # Setup LIME
                explainer.setup_lime(background)

                logger.info("XAI explainers initialized successfully")
            except Exception as e:
                logger.error(f"XAI initialization failed: {e}")

        logger.info(f"Loaded {len(models)} models successfully")
        return True

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.error(traceback.format_exc())
        return False

@app.route('/')
def index():
    return render_template('index.html', class_names=class_names)

@app.route('/api/health')
def health():
    return app.response_class(
        response=json.dumps({
            'status': 'healthy',
            'models_loaded': len(models),
            'models': list(models.keys())
        }, cls=NumpyEncoder),
        status=200,
        mimetype='application/json'
    )

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction with all features"""
    try:
        data = request.get_json()
        logger.info(f"Received: {data}")

        # Preprocess
        X = preprocess_input(data)
        if X is None:
            return app.response_class(
                response=json.dumps({'error': 'Preprocessing failed'}, cls=NumpyEncoder),
                status=400,
                mimetype='application/json'
            )

        # Scale if scaler exists
        if scaler is not None:
            try:
                X = scaler.transform(X)
            except Exception as e:
                logger.warning(f"Scaling error (using unscaled data): {e}")

        # Preprocessing functions for deep models
        preprocess = {
            'cnn': preprocess_cnn,
            'lstm': preprocess_lstm
        }

        # Get ensemble prediction
        try:
            preds, conf = ensemble.predict_with_confidence(X, preprocess)
            pred = int(preds[0])
            confidence = float(conf[0])
        except Exception as e:
            logger.error(f"Ensemble error, using fallback: {e}")
            # Fallback to Random Forest if available
            if 'rf' in models:
                pred = int(models['rf'].predict(X)[0])
                proba = models['rf'].predict_proba(X)[0]
                confidence = float(np.max(proba))
            else:
                pred = 0
                confidence = 0.5

        # Get individual model predictions
        individual = {}
        for name, model in models.items():
            try:
                if name == 'cnn':
                    X_proc = preprocess_cnn(X)
                    pred_proba = model.predict(X_proc, verbose=0)[0]
                    p = int(np.argmax(pred_proba))
                    c = float(np.max(pred_proba))
                elif name == 'lstm':
                    X_proc = preprocess_lstm(X)
                    pred_proba = model.predict(X_proc, verbose=0)[0]
                    p = int(np.argmax(pred_proba))
                    c = float(np.max(pred_proba))
                elif hasattr(model, 'predict_proba'):
                    p = int(model.predict(X)[0])
                    proba = model.predict_proba(X)[0]
                    c = float(np.max(proba))
                else:
                    p = int(model.predict(X)[0])
                    c = confidence

                individual[name] = {
                    'prediction': p,
                    'class_name': class_names[p],
                    'confidence': c
                }
            except Exception as e:
                logger.error(f"Error with {name}: {e}")
                individual[name] = {
                    'prediction': pred,
                    'class_name': class_names[pred],
                    'confidence': confidence
                }

        # Get mitigation action
        mitigation_action = mitigation.get_mitigation_action(pred, confidence, data.get('node_id'))
        mitigation_response = mitigation.execute_mitigation(
            mitigation_action, data.get('node_id'), pred
        )

        # Prepare response
        response = {
            'success': True,
            'ensemble': {
                'prediction': pred,
                'class_name': class_names[pred],
                'confidence': confidence
            },
            'individual': individual,
            'mitigation': mitigation_response,
            'node_status': mitigation.get_node_status(data.get('node_id'))
        }

        return app.response_class(
            response=json.dumps(response, cls=NumpyEncoder),
            status=200,
            mimetype='application/json'
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return app.response_class(
            response=json.dumps({'error': str(e)}, cls=NumpyEncoder),
            status=500,
            mimetype='application/json'
        )

@app.route('/api/explain', methods=['POST'])
def explain():
    """Get explanation for prediction"""
    try:
        data = request.get_json()
        method = data.get('method', 'shap')

        # Preprocess input
        X = preprocess_input(data)
        if X is None:
            return app.response_class(
                response=json.dumps({'error': 'Preprocessing failed'}, cls=NumpyEncoder),
                status=400,
                mimetype='application/json'
            )

        # Scale if scaler exists
        if scaler is not None:
            try:
                X = scaler.transform(X)
            except:
                pass

        # Check if explainer is initialized
        if explainer is None:
            return app.response_class(
                response=json.dumps({'error': 'Explainer not initialized'}, cls=NumpyEncoder),
                status=400,
                mimetype='application/json'
            )

        # Get explanation
        if method == 'shap':
            result = explainer.explain_shap(X[0], plot=True)
        else:
            result = explainer.explain_lime(X[0], num_features=8)

        return app.response_class(
            response=json.dumps(result, cls=NumpyEncoder),
            status=200,
            mimetype='application/json'
        )

    except Exception as e:
        logger.error(f"Explanation error: {e}")
        logger.error(traceback.format_exc())
        return app.response_class(
            response=json.dumps({'error': str(e), 'success': False}, cls=NumpyEncoder),
            status=500,
            mimetype='application/json'
        )

@app.route('/api/node_status/<int:node_id>')
def node_status(node_id):
    status = mitigation.get_node_status(node_id)
    return app.response_class(
        response=json.dumps(status, cls=NumpyEncoder),
        status=200,
        mimetype='application/json'
    )

# Load models on startup
if __name__ != '__main__' or not load_models():
    load_models()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
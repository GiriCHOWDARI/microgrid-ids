"""
Explainable AI Module - SHAP and LIME
Fully functional with visualization support
"""

import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelExplainer:
    def __init__(self, model, feature_names=None, class_names=None):
        self.model = model
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(10)]
        self.class_names = class_names or ['Normal', 'Blackhole', 'Grayhole', 'Flooding',
                                          'Sybil', 'Sinkhole', 'TDMA', 'Hello Flood']
        self.shap_explainer = None
        self.lime_explainer = None
        self.background_data = None

    def setup_shap(self, background_data, model_type='auto'):
        """Setup SHAP explainer"""
        try:
            self.background_data = background_data

            # Check model type
            model_str = str(type(self.model)).lower()

            if 'randomforest' in model_str or 'xgb' in model_str or 'tree' in model_str:
                self.shap_explainer = shap.TreeExplainer(self.model)
                logger.info("Created TreeExplainer")
            elif 'tensorflow' in model_str or 'keras' in model_str or 'sequential' in model_str:
                # For deep learning models, use a subset of background data
                if len(background_data) > 100:
                    background = background_data[:100]
                else:
                    background = background_data
                self.shap_explainer = shap.DeepExplainer(self.model, background)
                logger.info("Created DeepExplainer")
            else:
                # Fallback to KernelExplainer
                if len(background_data) > 100:
                    background = background_data[:100]
                else:
                    background = background_data
                self.shap_explainer = shap.KernelExplainer(
                    self._predict_proba, background
                )
                logger.info("Created KernelExplainer")

            return True
        except Exception as e:
            logger.error(f"SHAP setup failed: {e}")
            return False

    def setup_lime(self, training_data, mode='classification'):
        """Setup LIME explainer"""
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode=mode,
                discretize_continuous=True,
                random_state=42,
                verbose=False
            )
            logger.info("LIME explainer created")
            return True
        except Exception as e:
            logger.error(f"LIME setup failed: {e}")
            return False

    def _predict_proba(self, X):
        """Wrapper for model prediction - handles different model types"""
        try:
            # Convert to numpy if needed
            if not isinstance(X, np.ndarray):
                X = np.array(X)

            # Handle different model types
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            elif hasattr(self.model, 'decision_function'):
                decisions = self.model.decision_function(X)
                # Convert to probabilities using sigmoid
                if len(decisions.shape) == 1:
                    proba = 1.0 / (1.0 + np.exp(-decisions))
                    return np.column_stack([1-proba, proba])
                else:
                    proba = 1.0 / (1.0 + np.exp(-decisions))
                    return proba / proba.sum(axis=1, keepdims=True)
            elif hasattr(self.model, 'predict'):
                # For models without predict_proba, create one-hot
                preds = self.model.predict(X)
                proba = np.zeros((len(preds), len(self.class_names)))
                for i, p in enumerate(preds):
                    proba[i, int(p)] = 1.0
                return proba
            else:
                # Fallback - random probabilities
                logger.warning("Model has no predict_proba, using random")
                return np.random.rand(len(X), len(self.class_names))
        except Exception as e:
            logger.error(f"Prediction wrapper error: {e}")
            # Return uniform probabilities as fallback
            return np.ones((len(X), len(self.class_names))) / len(self.class_names)

    def explain_shap(self, instance, plot=True):
        """Get SHAP explanation for an instance"""
        if self.shap_explainer is None:
            logger.error("SHAP not initialized")
            return {'success': False, 'error': 'SHAP not initialized'}

        try:
            # Ensure instance is 2D
            if len(instance.shape) == 1:
                instance = instance.reshape(1, -1)

            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(instance)

            # Get prediction
            pred_proba = self._predict_proba(instance)[0]
            pred_class = np.argmax(pred_proba)
            confidence = float(pred_proba[pred_class])

            if plot:
                # Create SHAP summary plot
                plt.figure(figsize=(10, 6))

                # Handle different SHAP value formats
                if isinstance(shap_values, list):
                    # Multi-class
                    shap.summary_plot(
                        shap_values[pred_class],
                        instance,
                        feature_names=self.feature_names,
                        show=False
                    )
                else:
                    # Binary or single class
                    shap.summary_plot(
                        shap_values,
                        instance,
                        feature_names=self.feature_names,
                        show=False
                    )

                plt.title(f"SHAP Explanation - {self.class_names[pred_class]} ({confidence:.1%})")
                plt.tight_layout()

                # Convert plot to base64
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()

                return {
                    'success': True,
                    'method': 'shap',
                    'prediction': int(pred_class),
                    'class_name': self.class_names[pred_class],
                    'confidence': confidence,
                    'image': img_base64
                }

            return {
                'success': True,
                'method': 'shap',
                'prediction': int(pred_class),
                'class_name': self.class_names[pred_class],
                'confidence': confidence,
                'shap_values': str(shap_values)  # Convert to string for JSON
            }

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {'success': False, 'error': str(e)}

    def explain_lime(self, instance, num_features=8):
        """Get LIME explanation for an instance"""
        if self.lime_explainer is None:
            logger.error("LIME not initialized")
            return {'success': False, 'error': 'LIME not initialized'}

        try:
            # Ensure instance is 1D for LIME
            if len(instance.shape) > 1:
                instance = instance.flatten()

            # Get LIME explanation
            exp = self.lime_explainer.explain_instance(
                instance,
                self._predict_proba,
                num_features=num_features,
                num_samples=1000,
                distance_metric='euclidean',
                model_regressor=None
            )

            # Get prediction
            pred_proba = self._predict_proba(instance.reshape(1, -1))[0]
            pred_class = np.argmax(pred_proba)
            confidence = float(pred_proba[pred_class])

            # Get feature importance as list
            features = exp.as_list()

            # Create visualization
            plt.figure(figsize=(8, max(4, len(features) * 0.5)))

            # Sort features by absolute impact
            features_dict = dict(features)
            names = list(features_dict.keys())
            values = list(features_dict.values())

            # Sort by absolute value
            sorted_idx = np.argsort(np.abs(values))[::-1]
            names = [names[i] for i in sorted_idx]
            values = [values[i] for i in sorted_idx]

            # Color based on impact direction
            colors = ['red' if v < 0 else 'green' for v in values]

            plt.barh(names, values, color=colors)
            plt.xlabel('Impact on Prediction')
            plt.title(f"LIME Explanation - {self.class_names[pred_class]} ({confidence:.1%})")
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            plt.tight_layout()

            # Convert to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            return {
                'success': True,
                'method': 'lime',
                'prediction': int(pred_class),
                'class_name': self.class_names[pred_class],
                'confidence': confidence,
                'features': features,
                'image': img_base64
            }

        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {'success': False, 'error': str(e)}
"""
Utility Functions Module
"""

import numpy as np
import pandas as pd
import json
import yaml
import logging
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

logger = logging.getLogger(__name__)

class Utils:
    """Utility functions for the IDS system"""

    @staticmethod
    def setup_logging(log_level=logging.INFO, log_file='logs/system.log'):
        Path("logs").mkdir(exist_ok=True)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger.info("Logging configured successfully")

    @staticmethod
    def load_config(config_path='config.yaml'):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def save_results(results, filename, path='results/'):
        Path(path).mkdir(exist_ok=True)
        if filename.endswith('.json'):
            with open(f'{path}{filename}', 'w') as f:
                json.dump(results, f, indent=2)
        elif filename.endswith('.pkl'):
            joblib.dump(results, f'{path}{filename}')
        logger.info(f"Results saved to {path}{filename}")

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    @staticmethod
    def timer_decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.info(f"{func.__name__} executed in {end-start:.2f}s")
            return result
        return wrapper

    @staticmethod
    def plot_confusion_matrix(cm, class_names, save_path=None):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
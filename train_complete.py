"""
Complete Training Script - All 5 Models
Run: python train_complete.py
"""

import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score
from src.data_preprocessing import DataPreprocessor
from src.train_models import ModelTrainer
from src.passive_aggressive import PassiveAggressiveModel
from src.utils import Utils

# Setup
Utils.setup_logging(log_file='logs/training.log')
logger = logging.getLogger(__name__)

def train_all():
    print("="*60)
    print("🚀 MICROGRID IDS - COMPLETE TRAINING")
    print("="*60)

    # Create directories
    for d in ['data/raw', 'data/processed', 'models', 'logs']:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Step 1: Generate data
    logger.info("\n📊 Generating data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.generate_synthetic_wsn_data(n_samples=15000)
    df.to_csv('data/raw/wsn_data.csv', index=False)

    # Step 2: Preprocess
    logger.info("\n🔧 Preprocessing...")
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df)

    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Initialize trainer
    trainer = ModelTrainer(input_shape=(X_train.shape[1],))
    split = int(0.8 * len(X_train))

    results = {}

    # 1. CNN
    logger.info("\n" + "="*50)
    logger.info("1️⃣ Training CNN")
    logger.info("="*50)
    try:
        trainer.train_cnn(X_train[:split], y_train[:split],
                          X_train[split:], y_train[split:], epochs=15)
        X_test_cnn = X_test.reshape(-1, X_test.shape[1], 1)
        results['cnn'] = trainer.evaluate_model(
            trainer.models['cnn'], X_test_cnn, y_test, 'cnn'
        )
    except Exception as e:
        logger.error(f"CNN failed: {e}")

    # 2. LSTM - FIXED
    logger.info("\n" + "="*50)
    logger.info("2️⃣ Training LSTM")
    logger.info("="*50)
    try:
        # Reshape for LSTM: (samples, timesteps=1, features)
        X_train_lstm = X_train[:split].reshape(-1, 1, X_train.shape[1])
        X_val_lstm = X_train[split:].reshape(-1, 1, X_train.shape[1])

        # Train LSTM
        trainer.train_lstm(X_train_lstm, y_train[:split],
                          X_val_lstm, y_train[split:], epochs=15)

        if 'lstm' in trainer.models:
            X_test_lstm = X_test.reshape(-1, 1, X_test.shape[1])
            results['lstm'] = trainer.evaluate_model(
                trainer.models['lstm'], X_test_lstm, y_test, 'lstm'
            )
        else:
            logger.warning("LSTM model not available")
    except Exception as e:
        logger.error(f"LSTM failed: {e}")

    # 3. Random Forest
    logger.info("\n" + "="*50)
    logger.info("3️⃣ Training Random Forest")
    logger.info("="*50)
    trainer.train_random_forest(X_train, y_train)
    results['rf'] = trainer.evaluate_model(
        trainer.models['random_forest'], X_test, y_test, 'rf'
    )

    # 4. XGBoost
    logger.info("\n" + "="*50)
    logger.info("4️⃣ Training XGBoost")
    logger.info("="*50)
    trainer.train_xgboost(X_train, y_train)
    results['xgb'] = trainer.evaluate_model(
        trainer.models['xgboost'], X_test, y_test, 'xgb'
    )

    # 5. Passive-Aggressive
    logger.info("\n" + "="*50)
    logger.info("5️⃣ Training Passive-Aggressive")
    logger.info("="*50)
    pac = PassiveAggressiveModel()
    pac.train(X_train, y_train)
    pac.save('models/pac_model.pkl')

    y_pred = pac.predict(X_test)
    results['pac'] = {'accuracy': accuracy_score(y_test, y_pred)}
    logger.info(f"PAC Accuracy: {results['pac']['accuracy']:.4f}")

    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 TRAINING SUMMARY")
    logger.info("="*50)
    for name, metrics in results.items():
        if 'accuracy' in metrics:
            logger.info(f"{name:15}: {metrics['accuracy']:.4f}")

    # Save results - FIXED: now method exists
    trainer.save_evaluation_results(results, 'complete_results.json')
    logger.info("\n✅ Training complete!")

if __name__ == "__main__":
    train_all()
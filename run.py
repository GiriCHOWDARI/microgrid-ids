#!/usr/bin/env python3
"""
Main entry point - Run the application
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.app import app
from src.utils import Utils

# Setup logging
Utils.setup_logging()

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     Microgrid IDS - Cyber Attack Detection System        ║
    ║         with Explainable AI & Mitigation                 ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Check if models exist
    model_files = ['models/cnn_model.h5', 'models/rf_model.pkl',
                   'models/xgb_model.pkl', 'models/pac_model.pkl']

    if not all(Path(f).exists() for f in model_files):
        print("\n⚠️  Models not found. Running training first...\n")
        import train_complete

        train_complete.train_all()

    print("\n" + "=" * 50)
    print("🌐 Web Interface: http://localhost:5000")
    print("📝 API: http://localhost:5000/api/health")
    print("=" * 50 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
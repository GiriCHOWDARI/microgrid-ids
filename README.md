# Microgrid IDS - Cyber Attack Detection System for Wireless Sensor Networks

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.13-orange)
![License](https://img.shields.io/badge/license-MIT-yellow)

## 📋 Overview

A comprehensive machine learning-based Intrusion Detection System (IDS) for detecting cyber attacks in microgrid Wireless Sensor Networks (WSNs). The system uses a hybrid ensemble of CNN, LSTM, Random Forest, and XGBoost models to detect 7 different types of attacks with high accuracy and low false positives.

### Key Features

- 🔍 **Multi-model Detection**: CNN, LSTM, Random Forest, XGBoost ensemble
- 🎯 **7 Attack Types**: Blackhole, Grayhole, Flooding, Sybil, Sinkhole, TDMA, Hello Flood
- 🤖 **Explainable AI**: SHAP and LIME integration for interpretability
- 🛡️ **Automated Mitigation**: Node isolation, rate limiting, traffic rerouting
- 🌐 **Real-time Web Interface**: Flask-based dashboard for monitoring
- 📊 **Performance Analytics**: Comprehensive metrics and visualizations
- 🔧 **Edge Deployment**: Lightweight models suitable for Raspberry Pi

## 🏗️ System Architecture
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Data Layer │ -> │ Detection Layer│ -> │ Mitigation Layer│
│ - Sensors │ │ - CNN/LSTM │ │ - Node Isolation│
│ - Network Logs │ │ - Random Forest│ │ - Rate Limiting │
│ - Features │ │ - XGBoost │ │ - Traffic Reroute│
└─────────────────┘ └─────────────────┘ └─────────────────┘
│ │
↓ ↓
┌─────────────────┐ ┌─────────────────┐
│ Explainability │ │ Web Interface │
│ - SHAP/LIME │ │ - Dashboard │
│ - Visualizations│ │ - API │
└─────────────────┘ └─────────────────┘

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/microgrid-ids.git
cd microgrid-ids
# ⚡ Microgrid IDS

## AI-Based Cyber Attack Detection & Mitigation System for Wireless Sensor Networks in Smart Microgrids

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Flask](https://img.shields.io/badge/Flask-WebApp-black)
![Machine Learning](https://img.shields.io/badge/MachineLearning-IntrusionDetection-green)
![XAI](https://img.shields.io/badge/XAI-SHAP%20%7C%20LIME-purple)
![License](https://img.shields.io/badge/License-MIT-red)

---

# 📌 Project Overview

Microgrid IDS is a real-time AI-powered Intrusion Detection and Mitigation System (IDS) designed to secure Wireless Sensor Networks (WSNs) used in modern smart microgrid infrastructures.

The project combines Machine Learning, Deep Learning, Explainable AI (XAI), and Automated Mitigation Logic to detect sophisticated cyber-attacks affecting distributed energy systems.

Unlike traditional signature-based IDS solutions, this system uses a hybrid ensemble learning architecture capable of detecting both known and unknown attack patterns in real time.

The system is lightweight, scalable, explainable, and deployable on edge devices such as Raspberry Pi.

---

# 🎯 Problem Statement

Modern microgrids rely heavily on Wireless Sensor Networks (WSNs) for:

* Energy monitoring
* Load balancing
* Fault detection
* Distributed communication
* Smart energy management

However, WSNs are vulnerable to multiple cyber threats due to:

* Open wireless communication
* Resource-constrained sensor nodes
* Weak authentication mechanisms
* Dynamic network topology

Traditional IDS systems fail to:

* Detect evolving attacks
* Support real-time decision-making
* Provide explainable predictions
* Execute automated mitigation actions

This project addresses these challenges through a hybrid ML-based cybersecurity framework.

---

# 🚀 Key Features

## 🔍 Real-Time Intrusion Detection

* Real-time cyber attack classification
* Lightweight edge-compatible architecture
* Ensemble-based detection pipeline
* Reduced false positives and false negatives

---

## 🧠 Hybrid AI Models

The IDS integrates multiple ML/DL models:

| Model              | Purpose                           |
| ------------------ | --------------------------------- |
| CNN                | Spatial anomaly detection         |
| LSTM               | Sequential behavior analysis      |
| Random Forest      | Structured traffic classification |
| XGBoost            | Optimized ensemble prediction     |
| Passive-Aggressive | Adaptive online learning          |

---

## 🛡️ Supported Cyber Attacks

The system detects:

* Blackhole Attack
* Grayhole Attack
* Flooding Attack
* Sybil Attack
* Sinkhole Attack
* Hello Flood Attack
* Replay Attack
* TDMA Exploit
* DoS Attack
* Spoofing Attack

---

# ⚙️ Automated Mitigation System

Once an attack is detected, the system automatically performs mitigation actions.

| Attack Type          | Mitigation Strategy        |
| -------------------- | -------------------------- |
| Blackhole / Grayhole | Node isolation & rerouting |
| Flooding / DoS       | Rate limiting              |
| TDMA / Replay        | Timestamp validation       |
| Sybil / Spoofing     | Identity verification      |
| Sinkhole             | Alternate routing          |

---

# 🧠 Explainable AI (XAI)

To improve transparency and operator trust, the project integrates:

## SHAP

* Global feature importance
* Prediction explanation
* Attack reasoning visualization

## LIME

* Local prediction interpretation
* Instance-level explanation
* Feature contribution analysis

This helps operators understand why a node was classified as malicious.

---

# 🏗️ System Architecture

```text
┌─────────────────────────┐
│ Wireless Sensor Nodes   │
│ & Simulated Traffic     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Data Collection Layer   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Preprocessing Layer     │
│ • Cleaning              │
│ • Normalization         │
│ • Feature Engineering   │
│ • SMOTE Balancing       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Detection Layer         │
│ CNN + LSTM + RF + XGB   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Ensemble Decision       │
│ Majority Voting         │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Explainability Layer    │
│ SHAP / LIME             │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Mitigation Engine       │
│ Isolation / Alerts      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Flask Dashboard         │
│ Real-Time Visualization │
└─────────────────────────┘
```

---

# 📂 Project Structure

```bash
microgrid-ids/
│
├── app/                     # Flask web application
├── data/                    # Dataset files
├── logs/                    # Detection logs
├── models/                  # Saved ML/DL models
├── notebooks/               # Research notebooks
├── src/                     # Core source code
├── tests/                   # Unit tests
│
├── requirements.txt         # Dependencies
├── config.yaml              # Configuration settings
├── run.py                   # Main application
├── simple_app.py            # Lightweight app version
├── train_complete.py        # Training pipeline
└── README.md
```

---

# 🔬 Methodology

The project follows a modular machine learning pipeline:

1. Synthetic WSN traffic generation
2. Data preprocessing and normalization
3. Feature engineering
4. Model training
5. Ensemble prediction
6. Explainable AI interpretation
7. Automated mitigation
8. Dashboard visualization

---

# 📊 Dataset Information

A synthetic dataset was generated to simulate microgrid wireless sensor traffic.

## Features Used

* Packet delay
* Hop count
* Energy consumption
* Packet loss rate
* Battery level
* Routing changes
* Data rate
* Node trust score
* Traffic behavior

The dataset contains both:

* Normal traffic
* Malicious attack traffic

---

# 🧪 Machine Learning Pipeline

## CNN

Used for:

* Spatial traffic analysis
* Pattern extraction
* Feature representation learning

## LSTM

Used for:

* Sequence learning
* Temporal anomaly detection
* Replay & TDMA attack detection

## Random Forest

Used for:

* Structured traffic classification
* Robust ensemble prediction
* Low-computation deployment

## XGBoost

Used for:

* High-accuracy boosting
* Handling imbalanced datasets
* Complex attack classification

## Passive-Aggressive Classifier

Used for:

* Adaptive learning
* Streaming attack detection
* Lightweight online updates

---

# 📈 Performance Evaluation

## Model Comparison

| Model              | Accuracy | Precision | Recall | F1-Score |
| ------------------ | -------- | --------- | ------ | -------- |
| CNN                | 0.68     | 0.71      | 0.68   | 0.69     |
| LSTM               | 0.69     | 0.73      | 0.69   | 0.71     |
| Random Forest      | 0.81     | 0.83      | 0.81   | 0.82     |
| XGBoost            | 0.84     | 0.86      | 0.84   | 0.85     |
| Passive-Aggressive | 0.66     | 0.64      | 0.66   | 0.65     |
| Hybrid Ensemble    | 0.85     | 0.87      | 0.85   | 0.86     |

---

# 📉 Evaluation Metrics

The system is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC Curve
* False Positive Rate
* False Negative Rate

---

# 🌐 Web Dashboard

The Flask-based dashboard provides:

* Real-time attack prediction
* Network traffic visualization
* Node status monitoring
* SHAP/LIME explanations
* Detection confidence scores
* Mitigation alerts
* Attack analytics

---

# ⚡ Installation Guide

# 📋 Prerequisites

Install the following:

* Python 3.8+
* pip
* Git

---

# 🔧 Setup

## Clone Repository

```bash
git clone https://github.com/GiriCHOWDARI/microgrid-ids.git
cd microgrid-ids
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

## Start Main Application

```bash
python run.py
```

## Run Lightweight Flask App

```bash
python simple_app.py
```

---

# 🏋️ Train Models

To train the ML models:

```bash
python train_complete.py
```

Training pipeline includes:

* Data preprocessing
* Feature scaling
* SMOTE balancing
* Model training
* Evaluation
* Model saving

---

# 📐 UML & Design Diagrams

The project documentation includes:

* System Architecture Diagram
* Use Case Diagram
* Class Diagram
* Sequence Diagram
* Deployment Diagram
* Activity Diagram
* DFD Diagram
* Workflow Diagram
* Mitigation Decision Logic

---

# 🛠️ Technologies Used

| Category        | Technologies          |
| --------------- | --------------------- |
| Programming     | Python                |
| Backend         | Flask                 |
| Deep Learning   | TensorFlow, Keras     |
| ML Frameworks   | Scikit-learn, XGBoost |
| Explainability  | SHAP, LIME            |
| Data Processing | Pandas, NumPy         |
| Visualization   | Matplotlib, Plotly    |
| Deployment      | Flask, Edge Devices   |

---

# 🔐 Security Objectives

The system aims to:

* Detect malicious sensor activity
* Secure microgrid communications
* Minimize false alarms
* Enable autonomous mitigation
* Improve smart grid resilience
* Support edge deployment

---

# 📚 Applications

This project is suitable for:

* Smart Grid Security Research
* Wireless Sensor Network Security
* IoT Intrusion Detection
* AI-based Cybersecurity Systems
* Academic Research Projects
* Edge AI Security Applications

---

# 🚀 Future Enhancements

Future improvements include:

* Federated Learning
* Blockchain-based Authentication
* TinyML Deployment
* Cloud Monitoring System
* Kubernetes Deployment
* Real Packet Capture Integration
* Adversarial Attack Defense

---

# 🤝 Contributing

Contributions are welcome.

## Contribution Steps

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push branch
5. Open Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

## Giri Chowdari

AI | Cybersecurity | Machine Learning

### GitHub

[https://github.com/GiriCHOWDARI](https://github.com/GiriCHOWDARI)

---

# ⭐ Support

If you found this project useful:

* ⭐ Star the repository
* 🍴 Fork the project
* 📢 Share with others

---

# 🔗 Repository Link

[https://github.com/GiriCHOWDARI/microgrid-ids](https://github.com/GiriCHOWDARI/microgrid-ids)

---

# 📄 Documentation Reference

This README was prepared based on:

* Full academic project documentation
* System architecture diagrams
* ML methodology and evaluation report
* Flask deployment workflow
* Source code implementation

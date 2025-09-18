# 🫀 Federated Heart Disease Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/YashRajcode04/Federated-Heart-Disease-Prediction)](https://github.com/YashRajcode04/Federated-Heart-Disease-Prediction)
[![GitHub Issues](https://img.shields.io/github/issues/YashRajcode04/Federated-Heart-Disease-Prediction)](https://github.com/YashRajcode04/Federated-Heart-Disease-Prediction/issues)

This project implements **federated learning (FL) algorithms** to predict heart disease using a real-world dataset. It demonstrates **privacy-preserving collaborative training** across multiple simulated clients while evaluating model performance under **non-IID data distributions**.  

Federated learning allows multiple clients to train a shared model **without sharing raw data**, making it ideal for sensitive healthcare applications.

---

## 🔹 Features

- **Multiple Federated Learning Algorithms:**
  - **FedAvg** – Standard federated averaging
  - **FedProx** – Incorporates proximal regularization for heterogeneous clients
  - **FedSGD** – Single-step gradient aggregation
  - **FedPer** – Personalized layers for each client
- **Non-IID Data Handling:** Simulates client data heterogeneity using Dirichlet distribution
- **Evaluation Metrics:**
  - AUROC
  - F1 Score
  - Accuracy
  - Precision & Recall
- **Visualization:** Training curves and final performance comparison plots for all algorithms

---

## 📂 Repository Structure

├── data/
│ └── heart_2022_no_nans.csv # Heart disease dataset (~78 MB)
├── fl_algorithms.py # Federated learning utilities & model definitions
├── fl_heart.py # Main training & evaluation script
├── LICENSE
├── README.md
└── requirements.txt


> **Note:** The dataset is large (~78 MB). Consider using [Git LFS](https://git-lfs.github.com/) for version control.

---

## ⚙️ Installation

### Clone the repository

git clone https://github.com/YashRajcode04/Federated-Heart-Disease-Prediction.git
cd Federated-Heart-Disease-Prediction

### Create and activate a virtual environment

python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

### Install dependencies

pip install -r requirements.txt

---

## 🏃 Running the Project

To start federated learning experiments:

python fl_heart.py

Simulates multiple clients and trains models using different FL algorithms.
Generates training curves and final performance comparison plots.
Hyperparameters and client configurations can be modified directly in fl_heart.py

---

## 📈 Results & Visualizations

Training curves per client and global model
Comparison of AUROC, F1, Accuracy, Precision, and Recall across all algorithms
Insights into how personalized models (FedPer) perform on heterogeneous data

---

## ⚖️ License

This project is licensed under the MIT License

If you want, I can also **add a “Getting Started” section with sample plots/screenshots and example outputs**, which makes the README look more professional and GitHub-ready.  
Do you want me to do that next?

---

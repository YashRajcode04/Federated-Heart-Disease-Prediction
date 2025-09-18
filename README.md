# 🫀 Federated Heart Disease Prediction

This project implements **federated learning algorithms** (FedAvg, FedProx, FedSGD, FedPer) to predict heart disease using a real-world dataset.  
It demonstrates **privacy-preserving collaborative training** across multiple simulated clients while evaluating model performance.

---

## 🔹 Features

- Multiple federated learning algorithms:
  - **FedAvg** – standard federated averaging
  - **FedProx** – with proximal regularization
  - **FedSGD** – single-step gradient aggregation
  - **FedPer** – personalized layers per client
- Handles **non-IID client data** using Dirichlet distribution
- Evaluates models using:
  - AUROC
  - F1 Score
  - Accuracy
  - Precision & Recall
- Produces **training curves** and **final performance comparison plots**

---

## 📂 Repository Structure

.
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

1. Clone the repository:

git clone https://github.com/YashRajcode04/Federated-Heart-Disease-Prediction.git
cd Federated-Heart-Disease-Prediction


Create a virtual environment:

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

Install dependencies:

pip install -r requirements.txt

🏃 How to Run

Run the main script to start federated learning experiments:
python fl_heart.py

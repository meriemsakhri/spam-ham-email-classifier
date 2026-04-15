# 📧 Spam / Ham Email Classifier

> Binary email classification pipeline using NLP and Machine Learning.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](REMPLACE_PAR_TON_LIEN_COLAB)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

---

## 📌 Project Overview

This project builds a complete machine learning pipeline to classify emails
as **spam (1)** or **ham (0)** using Natural Language Processing techniques.

Two models are trained and compared:
- Logistic Regression (tuned with GridSearchCV)
- Random Forest (200 estimators, class_weight balanced)

---

## 📁 Repository Structure
spam-ham-email-classifier/
│
├── Projet_FULL_code.ipynb   # Main notebook (full pipeline)
├── spam_ham_dataset.csv     # Dataset (raw emails with labels)
├── README.md                # Project documentation
└── .gitignore               # Python gitignore

---

## ⚙️ Pipeline Steps

| Step | Description |
|------|-------------|
| 1. Ingestion | Load CSV, clean columns, check missing values |
| 2. EDA | Class distribution, email length analysis, top words |
| 3. Preprocessing | Tokenization, stopword removal |
| 4. Vectorization | TF-IDF (max 20,000 features, unigrams + bigrams) |
| 5. Splitting | Stratified 3-way split: 72% train / 13% val / 15% test |
| 6. Modeling | Logistic Regression + GridSearchCV, Random Forest |
| 7. Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix |
| 8. Explainability | Feature importances (Random Forest) |

---

## 📊 Results Summary

| Model | Accuracy | F1 (Spam) |
|-------|----------|-----------|
| Logistic Regression | ~0.98 | ~0.97 |
| Random Forest | ~0.98 | ~0.97 |

> Final scores evaluated on the held-out test set (used only once).

---

## 🛠️ Technologies Used

- Python 3.10
- pandas, numpy
- matplotlib, seaborn
- nltk (tokenization, stopwords)
- scikit-learn (TF-IDF, models, metrics, GridSearchCV)

---

## 🚀 How to Run

### Option 1 — Google Colab (recommended)
Click the badge at the top of this README.

### Option 2 — Local
```bash
git clone https://github.com/meriemsakhri/spam-ham-email-classifier.git
cd spam-ham-email-classifier
pip install pandas numpy matplotlib seaborn nltk scikit-learn
jupyter notebook Projet_FULL_code.ipynb
```

---

## 👤 Author

** MERIEM SAKHRI**  
Student — Ecole Polytechnique de Sousse  
[github.com/meriemsakhri](https://github.com/meriemsakhri)

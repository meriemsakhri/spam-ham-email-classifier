# 📧 Spam / Ham Email Classifier

> Binary email classification pipeline using NLP and Machine Learning.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1slbz4rU3mWLy1vRILFI3DSVOHyiQcoYM?usp=sharing)
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

```
spam-ham-email-classifier/
│
├── Projet_FULL_code.ipynb   # Main notebook (full pipeline)
├── spam_ham_dataset.csv     # Dataset (raw emails with labels)
├── requirements.txt         # Python dependencies (pinned versions)
├── README.md                # Project documentation
└── .gitignore               # Python gitignore
```

---

## ⚙️ Pipeline Steps

| Step | Description |
|------|-------------|
| 1. Ingestion | Load CSV, clean columns, check missing values |
| 2. EDA | Class distribution, email length analysis, top words |
| 3. Preprocessing | Tokenization, stopword removal |
| 4. Vectorization | TF-IDF (max 20,000 features, unigrams + bigrams) |
| 5. Splitting | Stratified 3-way split: **70% train / 15% val / 15% test** |
| 6. Modeling | Logistic Regression + GridSearchCV, Random Forest |
| 7. Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix |
| 8. Explainability | Feature importances (Random Forest) |

---

## 📊 Results Summary

Final scores on the held-out **test set** (opened only once):

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1 (Spam) | FP | FN |
|-------|:--------:|:----------------:|:-------------:|:---------:|:--:|:--:|
| **Logistic Regression** | **99.10 %** | **0.978** | 0.991 | **0.985** | **5** | 2 |
| Random Forest | 98.20 % | 0.949 | 0.991 | 0.970 | 12 | 2 |

> 🏆 **Winner: Logistic Regression** — higher F1, fewer false positives, same recall.
> Simpler, faster, and more interpretable — ideal for production security deployment.

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
pip install -r requirements.txt
jupyter notebook Projet_FULL_code.ipynb
```

---

## 👤 Author

**MERIEM SAKHRI**
Student — Ecole Polytechnique de Sousse
[github.com/meriemsakhri](https://github.com/meriemsakhri)

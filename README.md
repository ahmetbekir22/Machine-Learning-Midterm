Güzel fikir! Aşağıda tablo formatında güncellenmiş haliyle veriyorum. Bu haliyle hem daha profesyonel hem de okunabilir:

---

# Machine Learning Midterm Project - SE3007

This repository contains the implementation and results of the midterm project for the **Introduction to Machine Learning (SE3007)** course at Muğla Sıtkı Koçman University. The project explores key machine learning techniques and their practical applications through a series of structured tasks.

## 📚 Project Tasks

### 1. Missing Value Imputation

| Technique             | MSE   |
| --------------------- | ----- |
| Original Data         | 0.031 |
| Random Imputation     | 0.055 |
| Regression Imputation | 0.030 |

---

### 2. Model Training on Imputed Data

* **Model:** MLP Regressor (1 Hidden Layer, 10 Neurons, ReLU, Adam Optimizer, 500 Iterations)
* **Preprocessing:** StandardScaler

| Dataset               | MSE   |
| --------------------- | ----- |
| Original Data         | 0.031 |
| Random Imputation     | 0.055 |
| Regression Imputation | 0.030 |

---

### 3. Image Reconstruction (Dimensionality Reduction & Autoencoder)

* **Dataset:** MNIST (Sampled to 1000 images)
* **Techniques:**

  * PCA: Reduced features from 784 to 3.
  * MLP Regressor: (Hidden Layers: 10, 50) with varying epochs (10, 25, 500).

| Epochs | Reconstruction Quality |
| ------ | ---------------------- |
| 10     | Low                    |
| 25     | Medium                 |
| 500    | High                   |

---

### 4. Cluster Sampling

| Sampling Method | Accuracy | Training Time (ms) |
| --------------- | -------- | ------------------ |
| Original Data   | 0.888    | 2950               |
| Single-Stage    | 0.868    | 1750               |
| Double-Stage    | 0.840    | 749                |

---

### 5. Novelty Detection (Spam Classification)

* **Dataset:** SMS Spam Collection
* **Techniques:**

  * Text Preprocessing: Cleaning & Lowercasing
  * Feature Extraction: TF-IDF with N-Grams (1,2,3)
  * Model: SVM with RBF Kernel and Class Balancing

| Metric       | Value     |
| ------------ | --------- |
| TP           | 128       |
| TN           | 964       |
| FP           | 2         |
| FN           | 21        |
| **Accuracy** | **97.9%** |

---

## 📂 File Structure

```
├── Task1_2.py
├── Task3.py
├── Task4.py
├── Task5.py
├── MidtermReport.pdf
```

---

## ⚠️ Important Notice

This repository was created as part of the **SE3007: Introduction to Machine Learning** course, instructed by **Dr. Selim Yılmaz**.

This codebase is intended **only for educational purposes**. Future students from the same department may refer to this work, but **direct usage or submission without proper understanding will be considered academic dishonesty**. Always prioritize learning over copying! 

---


# README

## Overview
This repository provides an implementation of a **two-stage hierarchical machine learning model** to classify phenotypes from large-scale plasma proteomics data. It includes:

- Data preprocessing & quality control
- Borderline-SMOTE oversampling
- Two-layer hierarchical classification
- Permutation feature importance analysis
- Statistical testing and FDR correction

All machine-learning workflows are implemented using **Python 3.11.11** and **scikit-learn 1.4.2**.

---

## 1. Data Collection & Measurement Summary (for context)

Plasma samples from **1477 treatment-naïve patients** diagnosed in the U-CAN cohort (Uppsala, Sweden) were collected before clinical treatment. Samples were processed and archived at −80 °C for long-term storage.

Protein abundance for **3,879 total samples** was measured using **Olink Explore PEA (1536-plex)**, targeting **1,462 unique proteins** across four panels:  
- Cardiometabolic  
- Inflammation  
- Oncology  
- Neurology  

Olink internal/external controls were used to normalize data into NPX (log2) units. Features failing QC were removed.

> Only high-level description provided here — raw assays and metadata processing are assumed to be complete prior to modeling.

---

## 2. Mathematical Formulation of the Hierarchical Model

Given dataset  
\[
\{(x_i, y_i)\}_{i=1}^N, \quad x_i \in \mathbb{R}^n, \quad y_i \in \{1,\dots,M\}
\]

We first partition phenotype labels \(y_i\) into **K super-categories** forming disjoint equivalence classes \(\bar{y}_i\).

### Stage-1 model  
Trained on original labels \((x_i, y_i)\) to predict coarse category.

### Stage-2 models  
For each category \(k \in \{1,\dots,K\}\), a second-layer classifier \(g_k\) is trained using only samples from that category.

### Final prediction  
\[
P(X) = \sum_{j=1}^{K} P_f(X) \cdot P_{g_j}(X_j \mid f(X)) 
\]

If \(f\) is highly accurate (\(P(f(x_i) = y_i) \approx 1\)), this simplifies to:
\[
P(X) = P_f(X) \cdot P_{g_{f(X)}}(X \mid f(X))
\]

All models \(f, g_j \in \mathcal{F}\), where \(\mathcal{F}\) includes models from scikit-learn.

---

## 3. Two-Stage Hierarchical Classification

### 3.1 Candidate Models

20 scikit-learn classifiers were evaluated, including:
`AdaBoost`, `Bagging`, `BernoulliNB`,  
`DecisionTree`, `Dummy`, `ExtraTrees`,  
`GaussianNB`, `KNN`, `LabelPropagation`,  
`LGBM`, `LDA`, `LogisticRegression`,  
`MLP`, `NuSVC`, `Perceptron`,  
`RandomForest`, `RidgeClassifier`, `SGD`,  
`SVC`, `XGBoost`  

Model ranking used **mean F1 score** via **5-fold CV** on train-val split (70/30).

### 3.2 Final Chosen Models

| Stage | Category | Final model |
|-------|----------|-------------|
| 1     | All      | Perceptron |
| 2     | Blood        | ExtraTreesClassifier |
| 2     | Psychiatric  | SGDClassifier |
| 2     | Metabolic    | RidgeClassifier |
| 2     | Cancer       | LogisticRegression |

> ExtraTreesClassifier selected for blood although Ridge ranked slightly higher — performance difference negligible.

---

## 4. Handling Imbalanced Data: Borderline-SMOTE

We use **Borderline-SMOTE** from `imblearn`.  

For minority samples \(S_{\min} = \{x_{n_1}, \dots, x_{n_l}\}\):

1. Compute k nearest neighbors \(N_k(x)\) (Euclidean)
2. Count majority samples \(m_i\)
3. If \(k/2 \le m_i < k\):
   - Randomly select minority neighbor \(x_z\)
   - Generate synthetic sample  
\[
x_{\text{new}} = x_{n_i} + \lambda \cdot (x_z - x_{n_i}),\quad \lambda\sim U(0,1)
\]

Repeat until minority class is balanced.

---

## 5. Permutation Feature Importance

For trained model \(F\), test set \(X_{\text{test}} = \{x_1,\dots,x_k\}\):

1. Compute baseline score  
\[
s(F(X_{\text{test}}), y_{\text{test}})
\]

2. For feature \(i\), permute feature column 1000 times and compute score drops  
Approximation of:
\[
\frac{1}{n!}\sum_{\sigma_i} ( s(F(X), y) - s(F(\sigma_i(X)), y) )
\]

This approximates importance by random perturbation.

---

## 6. Statistical Analysis

- **FDR correction**: Benjamini–Hochberg  
- **t-test** assumed normality  
- **Bonferroni correction**:  
\[
p_{\text{corrected}} = p \times 1462
\]

Additional analyses incorporated STRING-DB.

---

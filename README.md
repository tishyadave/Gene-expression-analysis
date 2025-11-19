# Lung Cancer Detection Pipeline: Genomic Expression Analysis

## Project Overview

This project implements a **bioinformatics machine learning pipeline** to classify Lung Adenocarcinoma versus Normal tissue using microarray gene expression data (Dataset: **GSE10072**).

The core focus of this analysis is **statistical rigor**. High-dimensional genomic data (where *p* genes >> *n* samples) is prone to overfitting. This pipeline utilizes **Nested Cross-Validation** to prevent data leakage during feature selection, ensuring that the identified biomarkers and performance metrics are robust and reproducible.

## Key Features

* **Leakage-Free Validation:** Feature selection (`SelectKBest`) is performed strictly within training folds of a Stratified 5-Fold Cross-Validation loop.
* **Biomarker Discovery:** Identifies the top 15 most discriminative genes driving the classification.
* **Preprocessing Engine:** Handles variance filtering and Log2 transformation to normalize raw microarray intensity data.
* **Multi-Model Benchmarking:** Compares Random Forest, SVM (RBF), and MLP Neural Networks.

## Models 


**Neural Network**
**SVM (RBF)** 
**Random Forest**

### Visualizations

**1. Biological Signature (Heatmap)**
The top 15 genes show clear differential expression patterns, with distinct clusters for Normal vs. Tumor samples.
*(Note: Red indicates upregulation, Blue indicates downregulation)*

**2. Model Performance**
Benchmarking metrics showing high Sensitivity (Recall) across all classifiers, critical for medical diagnostics.

## 🛠️ Technical Methodology

1.  **Data Loading:** Parses raw GEO Series Matrix text files.
2.  **Variance Thresholding:** Removes constant features and the bottom 10% of low-variance genes to reduce noise.
3.  **Nested CV Pipeline:**
    * *Step 1:* Standard Scaling.
    * *Step 2:* Univariate Feature Selection (ANOVA F-value).
    * *Step 3:* Classifier Training.
4.  **Evaluation:** Metrics are aggregated across all validation folds.

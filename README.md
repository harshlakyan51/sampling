# **Credit Card Fraud Detection with Multiple Machine Learning Models**

## **Overview**

This project focuses on analyzing and detecting fraudulent credit card transactions using machine learning models. The dataset is imbalanced, with fraudulent transactions being much less frequent than non-fraudulent ones. To address this, various sampling techniques (such as SMOTE and other sampling methods) are applied to balance the dataset. Multiple machine learning models are trained, and the best-performing model for each subset of the data is identified.

## **Table of Contents**

- [Installation Instructions](#installation-instructions)
- [Project Description](#project-description)
- [Workflow](#workflow)
  - [Data Loading and Initial Exploration](#data-loading-and-initial-exploration)
  - [Class Distribution Analysis](#class-distribution-analysis)
  - [Missing Values Check](#missing-values-check)
  - [Handling Imbalanced Dataset](#handling-imbalanced-dataset-with-smote)
  - [Creating Samples](#creating-samples)
  - [Model Training and Evaluation](#model-training-and-evaluation)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)

## **Installation Instructions**

Before running the code, ensure you have Python 3.x installed. You'll need to install several Python libraries. Follow these steps:

1. Clone the repository or download the code.
2. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```
# Project Description
This project leverages machine learning algorithms to identify fraudulent credit card transactions. Given the imbalance in the dataset (fraudulent transactions are less frequent), various techniques such as SMOTE (Synthetic Minority Over-sampling Technique) are used to balance the dataset. Five different machine learning models are trained and evaluated on the balanced dataset to identify the best-performing models for fraud detection.

# Models Used:
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Support Vector Classifier (SVC)
K-Nearest Neighbors Classifier (KNN)
Key Techniques:
SMOTE for handling class imbalance.
Various sampling techniques to generate diverse subsets of the dataset for model training.

# Workflow
1. Data Loading and Initial Exploration
The dataset is loaded using pandas from a CSV file. We perform initial exploration:
```bash
data.head()
data.info()
data.describe()
```

data.head(): Displays the first 5 rows.
data.info(): Displays metadata (column names, data types, etc.).
data.describe(): Displays summary statistics of numerical columns.

# 2. Class Distribution Analysis
We examine the distribution of the target variable Class (fraud or non-fraud):
```bash
data["Class"].value_counts()
```

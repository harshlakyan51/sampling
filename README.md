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
## 1. Data Loading and Initial Exploration
The dataset is loaded using pandas from a CSV file. We perform initial exploration:
```bash
data.head()
data.info()
data.describe()
```

data.head(): Displays the first 5 rows.
data.info(): Displays metadata (column names, data types, etc.).
data.describe(): Displays summary statistics of numerical columns.

 ## 2. Class Distribution Analysis
We examine the distribution of the target variable Class (fraud or non-fraud):
```bash
data["Class"].value_counts()
```
## 3. Missing Values Check
We check for any missing values in the dataset:
```bash
missing_values = data.isnull().sum()
```
## 4. Handling Imbalanced Dataset with SMOTE
We apply SMOTE to create synthetic samples of the minority class (fraudulent transactions) and balance the dataset:
```bash
from imblearn.over_sampling import SMOTE
smote = SMOTE()
x_smote, y_smote = smote.fit_resample(x, y)
```
## 5. Creating Samples
We generate five samples using various sampling techniques:
```bash
# Simple Random Sampling
sample1 = balanced_data.iloc[np.random.choice(len(balanced_data), size=int(0.2 * len(balanced_data)), replace=False)]

# Stratified Sampling
strata = balanced_data.groupby('Class')
sample2 = strata.apply(lambda x: x.sample(int(0.2 * len(x)), random_state=2)).reset_index(drop=True)

# Systematic Sampling
k = len(balanced_data) // int(0.2 * len(balanced_data))
start = np.random.randint(0, k)
sample3 = balanced_data.iloc[start::k]

# Cluster Sampling
cluster_labels = np.arange(len(balanced_data)) % 5
balanced_data['Cluster'] = cluster_labels
selected_cluster = np.random.choice(5)
sample4 = balanced_data[balanced_data['Cluster'] == selected_cluster].drop('Cluster', axis=1)

# Bootstrapping
sample5 = balanced_data.iloc[np.random.choice(len(balanced_data), size=int(0.2 * len(balanced_data)), replace=True)]

```

## 6. Model Training and Evaluation
We train and evaluate multiple machine learning models using the generated samples:
```bash
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "k-NN": KNeighborsClassifier()
}

results = {}
samples = [sample1, sample2, sample3, sample4, sample5]

for model_name, model in models.items():
    results[model_name] = []
    for i, sample in enumerate(samples):
        X_sample = sample.drop('Class', axis=1)
        y_sample = sample['Class']
        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[model_name].append(accuracy)

results_df = pd.DataFrame(results, index=["Sample1", "Sample2", "Sample3", "Sample4", "Sample5"])
print(results_df)
results_df.to_csv("model_accuracies.csv")
```
# Requirements
Python 3.x
Libraries:
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
# Usage
Install the required libraries by running the following:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

##  Run the script to:

Load and inspect the dataset.
Preprocess the data (including visualizations).
Apply SMOTE to balance the dataset.
Generate 5 different samples using various sampling techniques.
Train 5 different machine learning models on each sample.
Evaluate the models and print the best-performing model for each sample.


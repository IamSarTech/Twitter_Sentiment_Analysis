# Twitter Sentiment Analysis Using Machine Learning

## Abstract

This project presents a machine learning–based sentiment analysis system designed to classify Twitter posts as either positive or negative. The model is trained on a large-scale dataset containing 1.6 million tweets and applies standard natural language processing techniques combined with a supervised learning algorithm. The objective is to automatically interpret public sentiment from textual social media data with reasonable accuracy and scalability.

---

## Introduction

With the rapid growth of social media platforms, vast amounts of unstructured textual data are generated daily. Analyzing sentiment from such data is essential for understanding public opinion, customer feedback, and social trends. This project focuses on building a sentiment classification model that processes raw tweet text and predicts the underlying sentiment using machine learning techniques.

---

## Dataset Description

- **Dataset Name:** Sentiment140  
- **Source:** Kaggle  
- **Total Samples:** 1,600,000 tweets  
- **Sentiment Labels:**
  - `0` → Negative sentiment  
  - `4` → Positive sentiment (converted to `1` during preprocessing)

The dataset is balanced, containing an equal number of positive and negative tweets.

---

## Tools and Technologies Used

- **Programming Language:** Python  
- **Development Environment:** Google Colab  
- **Libraries and Frameworks:**
  - NumPy
  - Pandas
  - Regular Expressions (`re`)
  - NLTK
  - Scikit-learn
  - Pickle

---

## Methodology

### 1. Data Acquisition
The dataset is downloaded programmatically using the Kaggle API and extracted from a compressed archive.

### 2. Data Preprocessing
The raw tweet text undergoes multiple preprocessing steps:
- Removal of URLs, numbers, and special characters
- Conversion to lowercase
- Stopword removal using NLTK
- Word stemming using Porter Stemmer

A new feature column named `stemmed_content` is created to store the cleaned text.

### 3. Feature Extraction
The processed text data is transformed into numerical form using TF-IDF (Term Frequency–Inverse Document Frequency) vectorization.

### 4. Dataset Splitting
The dataset is divided into training and testing subsets:
- 80% training data
- 20% testing data  

Stratified sampling is used to preserve class balance.

### 5. Model Training
A Logistic Regression classifier is trained on the TF-IDF features with a maximum of 1000 iterations to ensure convergence.

### 6. Model Evaluation
The trained model is evaluated using accuracy as the performance metric on both training and testing datasets.

---

## Results

| Dataset        | Accuracy |
|---------------|----------|
| Training Data | ~79.8%   |
| Testing Data  | ~77.6%   |

The results indicate that the model generalizes well and does not suffer from significant overfitting.

---

## Model Persistence

The trained model is serialized and saved using the Pickle library:


This allows the model to be reused for future predictions without retraining.

---

## Sample Prediction

```python
X_new = X_test[200]
prediction = model.predict(X_new)

if prediction[0] == 0:
    print("Negative tweet")
else:
    print("Positive tweet")

# **Natural Language Processing with Disaster Tweets**

## **Project Overview**
This project aims to classify tweets related to disasters using **Natural Language Processing (NLP)** and machine learning models. We implemented two models:
- **XGBoost** (Extreme Gradient Boosting)
- **Multinomial Naive Bayes** (MultinomialNB)

We preprocessed the text data, experimented with different feature extraction techniques, and optimized hyperparameters to achieve the best classification accuracy.

---

## **Dataset**
The dataset consists of tweets labeled as **disaster-related (1)** or **not disaster-related (0)**. The data was obtained from the Kaggle competition: [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started).

- **Total training samples**: 7,613
- **Total test samples**: 3,263
- **Features**:
  - `id`: Unique identifier for each tweet
  - `text`: The actual tweet content
  - `keyword`: Specific keyword related to the disaster (if available)
  - `location`: Tweet location (if available)
  - `target`: Binary classification label (1 = disaster-related, 0 = not disaster-related)

---

## **Preprocessing Steps**
To prepare the data for training, we applied the following **text preprocessing** techniques:

### **1. Data Cleaning**
✔️ Convert text to lowercase
✔️ Remove URLs, mentions (`@username`), and hashtags (`#keyword`)
✔️ Convert emojis into meaningful words
✔️ Replace common abbreviations (e.g., `omg` → `oh my god`)
✔️ Remove punctuation and special characters
✔️ Tokenize words and remove stopwords
✔️ Apply stemming to reduce words to their root form

### **2. Feature Engineering**
✔️ **TF-IDF (Term Frequency-Inverse Document Frequency):** Captures the importance of words in each tweet
✔️ **N-grams (unigrams, bigrams, trigrams):** Captures word sequences to enhance model performance
✔️ **Word Embeddings (GloVe/FastText):** Helps capture semantic relationships between words

---

## **Model Selection & Training**
We implemented **two machine learning models** and evaluated their performance:

### **1️⃣ Multinomial Naive Bayes (Baseline Model)**
MultinomialNB is a popular text classification algorithm that works well with **word frequency-based representations** (TF-IDF, CountVectorizer).

- **Hyperparameters:**
  - Alpha (`alpha=1.0`): Controls smoothing (Laplace smoothing)

- **Results:**
  - Kaggle leaderboard score: **0.76064**
  - Training Accuracy: **77.1%**
  - Validation Accuracy: **76.3%**
  
📌 **Pros:** Fast training and simple implementation
📌 **Cons:** Struggles with longer sequences and complex patterns

---

### **2️⃣ XGBoost (Optimized Model)**
XGBoost is a powerful gradient boosting algorithm known for its high accuracy and efficiency.

- **Hyperparameter Tuning:**
  - `n_estimators`: 100
  - `max_depth`: 4
  - `learning_rate`: 0.1
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  
- **Results:**
  - Kaggle leaderboard score: **0.78823**
  - Training Accuracy: **85.2%**
  - Validation Accuracy: **80.4%**
  
📌 **Pros:** Handles feature interactions well, robust against noisy data
📌 **Cons:** Computationally expensive compared to Naive Bayes

---

## **Model Comparison**
| Model | Training Accuracy | Validation Accuracy | Kaggle Score |
|--------|------------------|---------------------|--------------|
| Multinomial Naive Bayes | 77.1% | 76.3% | 0.76064 |
| XGBoost | 85.2% | 80.4% | 0.78823 |

### **Insights:**
- XGBoost **outperformed** MultinomialNB on all metrics, achieving a **higher Kaggle score**
- Naive Bayes, while fast, does not capture feature interactions as well as XGBoost
- Feature engineering played a significant role in improving both models

---

## **Future Improvements**
To further improve performance, we plan to:
✔️ **Experiment with deep learning models** (LSTMs, Transformers, BERT, RoBERTa)
✔️ **Use word embeddings** (GloVe, Word2Vec, FastText) instead of TF-IDF
✔️ **Perform more extensive hyperparameter tuning** with Optuna
✔️ **Incorporate external datasets** to improve generalization

---

## **Conclusion**
- **XGBoost significantly outperformed Multinomial Naive Bayes**, achieving **0.78823 Kaggle score**
- **Preprocessing and feature engineering played a critical role** in improving accuracy
- **Hyperparameter tuning improved model performance** and boosted results
- Future work will focus on **deep learning techniques** for even better classification accuracy

---

📌 **Kaggle Leaderboard Scores:**  
🔹 XGBoost: **0.78823**  
🔹 MultinomialNB: **0.76064**  

🚀 **Thank you for reading! Let's push for even better scores!** 🔥



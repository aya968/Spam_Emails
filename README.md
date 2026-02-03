# SMS Spam Detection Project

## Overview
This project implements a **Spam vs Ham SMS message classifier** using **Natural Language Processing (NLP)** and **Machine Learning**. The goal is to classify SMS messages as either "spam" or "ham" (not spam).  

The project includes:
- Data preprocessing and cleaning
- Feature extraction using TF-IDF
- Handling imbalanced dataset using SMOTE
- Training a Random Forest classifier
- Model evaluation with metrics
- Visualizations including word clouds and pie charts

---

## Dataset
The dataset contains **5,572 SMS messages** labeled as either `ham` or `spam`.  

**Columns:**
- `Category`: Label of the message (`ham` or `spam`)
- `Message`: Raw text of the SMS

**Example:**
| Category | Message |
|----------|---------|
| ham      | Ok lar... Joking wif u oni... |
| spam     | Free entry in 2 a wkly comp to win FA Cup final... |

The dataset is available from [Kaggle - Spam Text Message Classification](https://www.kaggle.com/team-ai/spam-text-message-classification).

---

## Libraries Used
```python
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px


Data Preprocessing

Remove duplicates to ensure unique SMS entries.

Text cleaning:

Lowercase text

Remove punctuation and numbers

Remove stopwords

Apply stemming using PorterStemmer

Tokenization: Convert text into tokens for feature extraction.

Example cleaned message:

Raw: Free entry in 2 a wkly comp to win FA Cup final...

Cleaned: free entri wkli comp win fa cup final tkt st m


Feature Extraction

Used TF-IDF Vectorizer to convert text into numerical features.

Limited to 3,000 features to reduce dimensionality.

Handling Imbalanced Data

Dataset is imbalanced: more ham messages than spam.

Applied SMOTE (Synthetic Minority Oversampling Technique) to balance classes.

Model Training

Split dataset into train (80%) and test (20%) sets.

Used Random Forest Classifier for classification.

Model Performance:

Accuracy: 99.2%

Precision, Recall, F1-score: ~0.99 for both classes

              precision    recall  f1-score   support
ham              0.99      0.99      0.99       895
spam             0.99      0.99      0.99       912


Visualization

Class Distribution: Pie chart showing proportion of spam vs ham.

Word Clouds:

Overall messages

Only spam messages

Only ham messages

These visualizations help in understanding frequent words and patterns in messages.

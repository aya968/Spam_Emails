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

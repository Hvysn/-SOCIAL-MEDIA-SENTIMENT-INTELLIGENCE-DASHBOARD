# ============================================================================
# INSTALL AND IMPORT LIBRARIES
# ============================================================================

# Uncomment the line below if running in Google Colab
# !pip install textblob nltk sqlalchemy pymysql

import pandas as pd
import numpy as np
import re
import warnings
import os
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Database Library
from sqlalchemy import create_engine

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

print("=" * 80)
print("CODINGPY - SOCIAL MEDIA SENTIMENT ANALYSIS")
print("=" * 80)
print("All libraries loaded successfully\n")

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================

print("[STEP 1: DATA LOADING]")
print("-" * 80)

DATASETS = [
    {'file': 'Twitter_Data.csv', 'source': 'Twitter'},
    {'file': 'Reddit_Data.csv', 'source': 'Reddit'},
]

if len(DATASETS) > 5:
    print("WARNING: Maximum 5 datasets allowed. Using only the first 5.")
    DATASETS = DATASETS[:5]

print(f"Configured to load {len(DATASETS)} dataset(s)\n")

dataframes = []

for i, dataset in enumerate(DATASETS, 1):
    file_path = dataset['file']
    source_name = dataset['source']

    try:
        if not os.path.exists(file_path):
            print(f"Dataset {i}: '{file_path}' not found - SKIPPING")
            continue

        df_temp = pd.read_csv(file_path)

        possible_text_columns = [
            'text', 'clean_text', 'content', 'tweet', 'post',
            'comment', 'clean_comment', 'message', 'body',
            'description', 'caption', 'status', 'title',
            'selftext', 'full_text'
        ]

        text_col_found = None

        for col in possible_text_columns:
            if col in df_temp.columns:
                text_col_found = col
                break

        if text_col_found is None:
            string_columns = df_temp.select_dtypes(include=['object']).columns
            max_avg_length = 0

            for col in string_columns:
                avg_length = df_temp[col].astype(str).str.len().mean()
                if avg_length > max_avg_length and avg_length > 20:
                    max_avg_length = avg_length
                    text_col_found = col

            if text_col_found:
                print(f"Auto-detected text column: {text_col_found}")

        if text_col_found and text_col_found != 'text':
            df_temp.rename(columns={text_col_found: 'text'}, inplace=True)

        if text_col_found is None:
            print(f"Dataset {i}: No text column found - SKIPPING")
            continue

        df_temp['source'] = source_name
        dataframes.append(df_temp)

        print(f"Dataset {i} loaded successfully")
        print(f"Source: {source_name}")
        print(f"Rows: {df_temp.shape[0]}\n")

    except Exception as e:
        print(f"Dataset {i}: Error loading file - {e}\n")

if len(dataframes) == 0:
    raise FileNotFoundError("No datasets available to process")

df = pd.concat(dataframes, ignore_index=True)

print("Combined dataset rows:", len(df))
print("\nSource Distribution:")
print(df['source'].value_counts())
print()

# ============================================================================
# STEP 2: TEXT CLEANING
# ============================================================================

print("[STEP 2: TEXT CLEANING]")
print("-" * 80)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]

    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)
df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)

print("Text cleaning completed")
print("Remaining rows:", len(df), "\n")

# ============================================================================
# STEP 3: SENTIMENT ANALYSIS
# ============================================================================

print("[STEP 3: SENTIMENT ANALYSIS]")
print("-" * 80)

def get_polarity(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0

def get_sentiment_label(score):
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

df['polarity'] = df['clean_text'].apply(get_polarity)
df['sentiment'] = df['polarity'].apply(get_sentiment_label)

print("Sentiment distribution:")
print(df['sentiment'].value_counts(), "\n")

# ============================================================================
# STEP 4: FINAL DATASET
# ============================================================================

final_df = df[['source', 'clean_text', 'polarity', 'sentiment']]
final_df.to_csv("Final_Data.csv", index=False)

print("Final dataset saved as Final_Data.csv\n")

# ============================================================================
# STEP 5: MACHINE LEARNING
# ============================================================================

print("[STEP 5: MACHINE LEARNING]")
print("-" * 80)

X = final_df['clean_text']
y = final_df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%\n")
print(classification_report(y_test, y_pred))

final_df['ml_prediction'] = model.predict(tfidf.transform(final_df['clean_text']))
final_df.to_csv("Final_Data.csv", index=False)

print("Updated dataset saved\n")
# ============================================================================
# STEP 6: EXPORT FINAL DATA TO MYSQL (NON-DISRUPTIVE)
# ============================================================================

from sqlalchemy import create_engine

print("[STEP 6: EXPORTING DATA TO MYSQL]")
print("-" * 80)

# MySQL connection string
engine = create_engine(
    "mysql+pymysql://root:7419@localhost/social_analytics"
)

# Export final dataframe
final_df.to_sql(
    name="final_data",
    con=engine,
    if_exists="append",   # SAFE: does not drop table
    index=False
)

print("Data successfully inserted into MySQL table: final_data\n")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("=" * 80)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 80)
print("Total Records:", len(final_df))
print("Sources:", final_df['source'].unique())
print("Output File: Final_Data.csv")
print("=" * 80)

import pandas as pd
import re
import string
from html import unescape
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(text):
    """ Clean and preprocess text data. """
    text = unescape(str(text))
    text = re.sub(r'http\S+|@\w+', '', text)
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

def train_text_model(df):
    """ Train a Logistic Regression model for emotion classification. """
    df['processed_content'] = df['content'].apply(preprocess_text)
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['processed_content'])
    y = df['sentiment']

    text_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    text_model.fit(X, y)
    
    return tfidf, text_model

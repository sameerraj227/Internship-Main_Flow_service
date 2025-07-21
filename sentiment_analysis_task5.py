
# Task 2: Sentiment Analysis using NLP and Logistic Regression

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Make sure to download required resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Sample dataset
data = {
    'Review Text': [
        "I love this product!", "Worst experience ever.", "Not bad at all",
        "Absolutely terrible", "Very good quality", "I hate this",
        "Excellent!", "Will never buy again", "Worth the money", "Total waste"
    ],
    'Sentiment': ['positive', 'negative', 'positive', 'negative', 'positive',
                  'negative', 'positive', 'negative', 'positive', 'negative']
}
df = pd.DataFrame(data)

# 1. Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()  # lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    words = nltk.word_tokenize(text)  # tokenize
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]  # remove stopwords & lemmatize
    return ' '.join(words)

df['Cleaned Text'] = df['Review Text'].apply(preprocess)

# 2. TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Cleaned Text'])
y = df['Sentiment']

# 3. Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Model Evaluation
y_pred = model.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Extra: show sample prediction
sample = ["This product is amazing", "I regret buying this"]
sample_cleaned = [preprocess(text) for text in sample]
sample_vectorized = vectorizer.transform(sample_cleaned)
sample_pred = model.predict(sample_vectorized)
print("\n--- Sample Prediction ---")
for txt, label in zip(sample, sample_pred):
    print(f"{txt} --> {label}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv('spam_Emails_data.csv', on_bad_lines='skip')
df = df.dropna(subset=['text', 'label'])

X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y)

vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 6969)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(C = 1.0, solver = 'liblinear', max_iter = 1000)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

print(classification_report(y_test, y_pred, target_names = ['ham', 'spam']))
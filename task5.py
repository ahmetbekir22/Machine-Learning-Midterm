from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate  # For LaTeX-like table formatting
import pandas as pd
import re

df = pd.read_csv('SMSSpamCollection', delimiter='\t', names=['label', 'message'])

# Preprocessing the message column: remove non-word characters
df['message'] = df['message'].apply(lambda x: re.sub(r'\W', ' ', x))  

# Convert all text in the 'message' column to lowercase 
df['message'] = df['message'].apply(lambda x: x.lower())

# Map labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Apply TF-IDF vectorization to convert text data into numerical features
# Using n-grams (1 to 3 words) and removing stop words
vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='word', stop_words='english', min_df=1)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Support Vector Classifier (SVC) model with an RBF kernel
clf = SVC(kernel='rbf', C=1.0, class_weight='balanced', gamma='scale', random_state=42)
clf.fit(X_train_tfidf, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test_tfidf)

# Calculate confusion matrix components
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = accuracy_score(y_test, y_pred)

# Format results into a table
metrics = [
    ["TP", tp],
    ["TN", tn],
    ["FP", fp],
    ["FN", fn],
    ["Accuracy", f"{accuracy:.3f}"]
]

# Display results as a table
print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="grid"))


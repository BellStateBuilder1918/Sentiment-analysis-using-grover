import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

def preprocess_text(text):
    """Preprocesses input text by cleaning, tokenizing, removing stopwords, and stemming."""
    if not isinstance(text, str):
        return ""  # Handle non-string inputs

    # Remove special characters, numbers, and retain only alphabets
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()  # Convert to lowercase and strip whitespace

    # Tokenize the text
    tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Apply stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

# Load the training dataset
train_data = pd.read_csv('train(f).csv')
train_data['text'] = train_data['text'].fillna('').astype(str).apply(preprocess_text)

# Feature extraction using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_data['text'])
y_train = train_data['label']

# Handle class imbalance
classes = [ 0, 1]  # Define the expected classes
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = {cls: weights[i] for i, cls in enumerate(classes)}

# Train a Logistic Regression model
model = LogisticRegression(class_weight=class_weights, max_iter=500)
model.fit(X_train, y_train)

# Load the evaluation dataset
eval_data = pd.read_csv('test(f).csv')
eval_data['text'] = eval_data['text'].fillna('').astype(str).apply(preprocess_text)
X_test = vectorizer.transform(eval_data['text'])
y_test = eval_data['label']

# Predict sentiments
predictions = model.predict(X_test)
eval_data['predicted_sentiment'] = predictions

# Evaluate model performance
accuracy = accuracy_score(y_test, predictions)
classification_rep = classification_report(y_test, predictions, target_names=['Negative', 'Neutral', 'Positive'])

# Save the updated evaluation dataset with predictions
eval_data.to_csv('sentiment_data_test_with_predictions.csv', index=False)

# Print results
print("Model Performance:")
print(f"Accuracy: {accuracy:.4f}\n")
print("Detailed Classification Report:")
print(classification_rep)

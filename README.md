# Quantum-Enhanced Sentiment Analysis
Comparative study between quantum enhanced model and classical models. The overall accuracy of the Logistic model was 82%, KKN (with cosine similarity) model was 75% and Decision trees (with single word features) is 80% while that of the quantum model was 90%.
## Overview
This project implements a **Quantum-Enhanced Sentiment Analysis Model**, leveraging **Grover's Algorithm for Quantum Feature Selection** and a **Random Forest Classifier** for sentiment classification. The model integrates classical **TF-IDF** and **Word2Vec** feature extraction with quantum feature selection to enhance text classification.

## Features
- **Data Preprocessing**: Cleans and tokenizes text data.
- **Feature Extraction**: Uses TF-IDF and Word2Vec embeddings.
- **Quantum Feature Selection**: Implements Grover’s algorithm to identify optimal features.
- **Machine Learning Classification**: Uses Random Forest Classifier for sentiment analysis.
- **Model Evaluation**: Computes accuracy and generates a classification report.

## Dependencies
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```
### Expected Output
- Quantum feature selection results.
- Model accuracy and classification report.
- Results saved in `sentiment_evaluation_results.csv`.

## How It Works
1. **Data Loading**: Reads `sentiment_data_train.csv` and `sentiment_data_test.csv`.
2. **Text Preprocessing**: Cleans and tokenizes the text.
3. **Feature Extraction**:
   - **TF-IDF**: Captures word importance.
   - **Word2Vec**: Captures word relationships.
4. **Quantum Feature Selection**:
   - Grover's algorithm identifies the most relevant features.
5. **Training & Prediction**:
   - Trains a Random Forest Classifier.
   - Predicts sentiment on test data.
6. **Evaluation**:
   - Computes accuracy and generates a classification report.
   - Saves results to CSV.

## Results
- The quantum feature selection process enhances the model by selecting optimal feature subsets.
- The classifier achieves a measurable accuracy improvement compared to traditional methods.


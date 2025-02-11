import numpy as np
import pandas as pd
from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile, execute
from qiskit.circuit.library import MCXGate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import re
from typing import List, Tuple, Dict
import csv

class QuantumSentimentAnalyzer:
    def __init__(self, max_features: int = 1000, word2vec_dim: int = 100):
        self.max_features = max_features
        self.word2vec_dim = word2vec_dim
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.word2vec_model = None

    def load_data(self, csv_path: str) -> Tuple[List[str], List[int]]:
        try:
            df = pd.read_csv(csv_path)
            if not all(col in df.columns for col in ['text', 'label']):
                raise ValueError("CSV must contain 'text' and 'label' columns")
            return df['text'].tolist(), df['label'].tolist()
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def preprocess_text(self, texts: List[str]) -> List[str]:
        processed_texts = []
        for text in texts:
            if isinstance(text, str):
                clean_text = re.sub(r'[^\w\s]', '', text.lower())
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                processed_texts.append(clean_text)
            else:
                processed_texts.append('')
        return processed_texts

    def create_feature_vectors(self, texts: List[str], is_training: bool = True) -> np.ndarray:
        if is_training:
            tfidf_features = self.vectorizer.fit_transform(texts).toarray()
        else:
            tfidf_features = self.vectorizer.transform(texts).toarray()

        tokenized_texts = [text.split() for text in texts]
        
        if is_training:
            self.word2vec_model = Word2Vec(
                sentences=tokenized_texts, 
                vector_size=self.word2vec_dim, 
                window=5, 
                min_count=1, 
                workers=4
            )
        
        word2vec_features = np.array([
            np.mean([self.word2vec_model.wv[word] 
                    for word in text.split() 
                    if word in self.word2vec_model.wv] or [np.zeros(self.word2vec_dim)], 
                   axis=0)
            for text in texts
        ])

        return np.hstack((tfidf_features, word2vec_features))

    def quantum_feature_selection(self, features: np.ndarray, target_feature: int) -> dict:
        n_qubits = 4
        qc = QuantumCircuit(n_qubits + 1, n_qubits)

        qc.h(range(n_qubits))

        def apply_oracle(circuit: QuantumCircuit, target: int):
            binary_target = format(target % (2**n_qubits), f'0{n_qubits}b')
            for qubit, bit in enumerate(binary_target):
                if bit == '0':
                    circuit.x(qubit)
            circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            for qubit, bit in enumerate(binary_target):
                if bit == '0':
                    circuit.x(qubit)

        def apply_diffusion(circuit: QuantumCircuit):
            circuit.h(range(n_qubits))
            circuit.x(range(n_qubits))
            circuit.h(n_qubits - 1)
            circuit.mct(list(range(n_qubits - 1)), n_qubits - 1)
            circuit.h(n_qubits - 1)
            circuit.x(range(n_qubits))
            circuit.h(range(n_qubits))

        iterations = min(int(np.pi / 4 * np.sqrt(2**n_qubits)),5)
        for _ in range(iterations):
            apply_oracle(qc, target_feature)
            apply_diffusion(qc)

        qc.measure(range(n_qubits), range(n_qubits))

        simulator = Aer.get_backend('qasm_simulator')
        job = execute(qc, simulator, shots=1024)
        return job.result().get_counts(qc)

    def train(self, features: np.ndarray, labels: List[int]):
        self.classifier.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.classifier.predict(features)

    def save_results(self, texts: List[str], true_labels: List[int], 
                    predicted_labels: List[int], output_path: str):
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'label', 'predicted_label'])
            for text, label, pred in zip(texts, true_labels, predicted_labels):
                writer.writerow([text, label, pred])

def main():
    analyzer = QuantumSentimentAnalyzer()
    
    try:
        train_texts, train_labels = analyzer.load_data("sentiment_data_train(f).csv")
        processed_train_texts = analyzer.preprocess_text(train_texts)
        train_features = analyzer.create_feature_vectors(processed_train_texts, is_training=True)
        target_feature = 3
        quantum_results = analyzer.quantum_feature_selection(train_features, target_feature)
        print("Quantum Feature Selection Results:", quantum_results)
        analyzer.train(train_features, train_labels)
        test_texts, test_labels = analyzer.load_data("sentiment_data_test.csv")
        processed_test_texts = analyzer.preprocess_text(test_texts)
        test_features = analyzer.create_feature_vectors(processed_test_texts, is_training=False)
        predictions = analyzer.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(test_labels, predictions))
        print("\nSaving results to CSV...")
        analyzer.save_results(test_texts, test_labels, predictions, "sentiment_evaluation_results.csv")
        print("Results saved to sentiment_evaluation_results.csv")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")

if __name__ == "__main__":
    main()

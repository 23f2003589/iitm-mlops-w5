import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import argparse

def load_original_model():
    model = joblib.load('model.joblib')
    
def poison_data(X, y, poison_percent=0.05):
    n_samples = len(X)
    n_poison = int(n_samples * poison_percent)    
    X_poisoned = X.copy()
    y_poisoned = y.copy()
    poison_indices = np.random.choice(n_samples, n_poison, replace=False)
    for idx in poison_indices:
        original_label = y_poisoned[idx]
        available_labels = [l for l in np.unique(y) if l != original_label]
        y_poisoned[idx] = np.random.choice(available_labels)
    return X_poisoned, y_poisoned

def train_and_evaluate(poison_percent=0.0, save_metrics=True):
    iris = load_iris()
    X, y = iris.data[:160], iris.target[:160]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if poison_percent > 0:
        X_train_poisoned, y_train_poisoned = poison_data(X_train, y_train, poison_percent)
    else:
        X_train_poisoned, y_train_poisoned = X_train, y_train
    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_poisoned, y_train_poisoned)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    if save_metrics:
        metrics = {
            "poison_percent": poison_percent,
            "accuracy": float(accuracy),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_poisoned_samples": int(len(X_train) * poison_percent)
        }
        
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    cm = confusion_matrix(y_test, y_pred)
    np.save('confusion_matrix.npy', cm)
    
    joblib.dump(model, 'model_poisoned.joblib')    
    return accuracy, model

def compare_models(poison_percent=0.05):
    original_model = load_original_model()
    iris = load_iris()
    X, y = iris.data[:160], iris.target[:160]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if original_model:
        y_pred_original = original_model.predict(X_test)
        original_accuracy = accuracy_score(y_test, y_pred_original)
        print(f"Original Model Accuracy: {original_accuracy:.4f}")
    poisoned_accuracy, _ = train_and_evaluate(poison_percent, save_metrics=True)
    if original_model:
        accuracy_drop = original_accuracy - poisoned_accuracy
        print(f"Accuracy Drop: {accuracy_drop:.4f} ({accuracy_drop*100:.2f}%)")

compare_models(poison_percent=0.05)
from sklearn import datasets
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib
import numpy as np
import time
import json
import os
from datetime import datetime

def train_model(model_type="DecisionTree", test_size=0.2, random_state=42, params={}):
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = list(iris.target_names)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Select and initialize the model based on model_type
    if model_type == "DecisionTree":
        model = tree.DecisionTreeClassifier(**params)
    elif model_type == "RandomForest":
        model = RandomForestClassifier(**params)
    elif model_type == "SVC":
        model = SVC(**params)
    elif model_type == "LogisticRegression":
        model = LogisticRegression(**params)
    elif model_type == "KNeighbors":
        model = KNeighborsClassifier(**params)
    else:
        # Default to Decision Tree if an invalid model type is provided
        model = tree.DecisionTreeClassifier()
    
    # Time the training
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for JSON serialization
    cr = classification_report(y_test, y_pred, target_names=target_names)
    
    # Save the model
    joblib.dump(model, "model.pkl")
    
    # Save model info
    model_info = {
        "model_type": model_type,
        "accuracy": accuracy,
        "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": params,
        "test_size": test_size,
        "random_state": random_state
    }
    
    with open("model_info.json", "w") as f:
        json.dump(model_info, f)
    
    # Return metrics
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": cr,
        "training_time": training_time,
        "model_type": model_type
    }

if __name__ == "__main__":
    # Default training with Decision Tree if script is run directly
    result = train_model()
    print(f"Model trained with accuracy: {result['accuracy']:.4f}")
    print(f"Model saved to model.pkl")
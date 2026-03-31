import os
import json
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Constants
NAME = "Manvith M"
ROLL_NO = "2022BCS0066"
EXPERIMENT_NAME = f"{ROLL_NO}_experiment"

# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

def load_data(version_path):
    df = pd.read_csv(version_path)
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y

def run_experiment(run_name, dataset_path, model_type, params, reduce_features=False):
    with mlflow.start_run(run_name=run_name):
        # Load dataset
        X, y = load_data(dataset_path)
        
        # Feature selection (select an arbitrary subset of features to satisfy the requirement)
        if reduce_features:
            selected_features = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash']
            X = X[selected_features]
            mlflow.log_param("num_features", len(selected_features))
            mlflow.log_param("selected_features", ",".join(selected_features))
        else:
            mlflow.log_param("num_features", X.shape[1])
            mlflow.log_param("selected_features", "all")
            
        # Log basic details
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if model_type == "RandomForest":
            # For Scikit-Learn RandomForest
            model = RandomForestClassifier(random_state=42, **params)
        elif model_type == "LogisticRegression":
            model = LogisticRegression(random_state=42, **params)
        else:
            raise ValueError("Unsupported model type")
            
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.sklearn.log_model(model, "model")
        
        import joblib
        joblib.dump(model, "model.pkl")
        
        return {
            "run_name": run_name,
            "accuracy": acc,
            "f1_score": f1
        }

if __name__ == "__main__":
    runs_plan = [
        {
            "run_name": "Run 1 - Base RF v1",
            "dataset_path": "data/v1/wine.csv",
            "model_type": "RandomForest",
            "params": {"n_estimators": 10},
            "reduce_features": False
        },
        {
            "run_name": "Run 2 - Tuned RF v1",
            "dataset_path": "data/v1/wine.csv",
            "model_type": "RandomForest",
            "params": {"n_estimators": 50, "max_depth": 5},
            "reduce_features": False
        },
        {
            "run_name": "Run 3 - Base RF v2",
            "dataset_path": "data/v2/wine.csv",
            "model_type": "RandomForest",
            "params": {"n_estimators": 10},
            "reduce_features": False
        },
        {
            "run_name": "Run 4 - Base RF v2 Reduced Features",
            "dataset_path": "data/v2/wine.csv",
            "model_type": "RandomForest",
            "params": {"n_estimators": 10},
            "reduce_features": True
        },
        {
            "run_name": "Run 5 - Logistic Regression v2 Reduced",
            "dataset_path": "data/v2/wine.csv",
            "model_type": "LogisticRegression",
            "params": {"max_iter": 1000},
            "reduce_features": True
        }
    ]
    
    results = []
    for run_config in runs_plan:
        print(f"Executing: {run_config['run_name']}")
        res = run_experiment(**run_config)
        results.append(res)
        
    metrics_output = {
        "Student Name": NAME,
        "Roll No": ROLL_NO,
        "Results": results
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=4)
        
    print("Training complete! Metrics saved to metrics.json.")

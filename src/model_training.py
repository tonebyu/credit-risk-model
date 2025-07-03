import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

def train_models(input_path):
    """
    Train and evaluate credit risk classification models with hyperparameter tuning.

    Args:
        input_path (str): Path to the input CSV dataset file.

    Returns:
        best_model: The trained model with the highest ROC-AUC score.

    Workflow:
        - Load dataset from CSV.
        - Extract datetime features from 'TransactionStartTime' if present.
        - Split dataset into train/test.
        - Define Logistic Regression and Random Forest models with parameter grids.
        - Perform GridSearchCV hyperparameter tuning on train data.
        - Evaluate models on test data using accuracy, precision, recall, F1, ROC-AUC.
        - Log all metrics and models in MLflow.
        - Register the best performing model in MLflow Model Registry.
    """
    df = pd.read_csv(input_path)

    # Optional: extract time features
    if 'TransactionStartTime' in df.columns:
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
        df.drop(columns=['TransactionStartTime'], inplace=True)

    X = df.drop(columns=['TransactionStartTime', 'is_high_risk', 'CustomerId'], errors='ignore')
    y = df['is_high_risk']

   

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42)
    }

    param_grid = {
        "LogisticRegression": {'C': [0.01, 0.1, 1, 10]},
        "RandomForest": {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
    }

    best_model = None
    best_score = 0

    for model_name, model in models.items():
        print(f"\n🔍 Training {model_name}...")
        mlflow.set_experiment("Credit-Risk-Modeling")

        with mlflow.start_run(run_name=f"{model_name}-experiment"):
            clf = GridSearchCV(model, param_grid[model_name], cv=3, scoring='roc_auc')
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_prob)
            }

            for key, val in metrics.items():
                mlflow.log_metric(key, val)

            mlflow.sklearn.log_model(clf.best_estimator_, model_name)

            print(classification_report(y_test, y_pred))

            if metrics['ROC-AUC'] > best_score:
                best_score = metrics['ROC-AUC']
                best_model = clf.best_estimator_

    print("\n✅ Best model:", best_model)
    

    
    return best_model

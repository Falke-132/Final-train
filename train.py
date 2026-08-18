import argparse
import os
import json
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train Phishing Detection Model")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the pre-cleaned CSV dataset")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Directory to save all artifacts (model, features, metrics)")
    return parser.parse_args()


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads a pre-cleaned dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Initializes and trains an ML pipeline using LightGBM.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.

    Returns:
        Pipeline: Trained sklearn Pipeline object.
    """
    best_params = {
        'learning_rate': 0.1,
        'max_depth': 10,
        'n_estimators': 200,
        'num_leaves': 31,
        'random_state': 42,
        'n_jobs': -1
    }

    pipeline = Pipeline([
        ('classifier', lgb.LGBMClassifier(**best_params))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluates the trained model on test data.

    Args:
        model (Pipeline): Trained sklearn Pipeline object.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target variable.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average='macro')),
        "roc_auc": float(roc_auc_score(y_test, y_proba))
    }


def save_artifacts(model: Pipeline, features: list, metrics: dict, output_dir: str):
    """
    Saves the trained model, feature list, and metrics to disk.

    Args:
        model (Pipeline): Trained sklearn Pipeline object.
        features (list): List of feature column names.
        metrics (dict): Dictionary of evaluation metrics.
        output_dir (str): Directory to save artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model, os.path.join(output_dir, 'phishing_pipeline.pkl'))
    joblib.dump(features, os.path.join(output_dir, 'features.pkl'))

    with open(os.path.join(output_dir, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)


def main():
    args = parse_args()

    print("1. Loading data...")
    df = load_dataset(args.input)

    X = df.drop('status', axis=1)
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("2. Training model...")
    model = train_model(X_train, y_train)

    print("3. Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")

    print(f"4. Saving artifacts to {args.output}...")
    save_artifacts(model, list(X.columns), metrics, args.output)
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script to train and save the ticket categorization and priority models.
Can be used with custom datasets to retrain the models.
"""

import os
import argparse
import pandas as pd
from ticket_assistant import TicketCategorizer, PriorityAssigner
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score


def train_models(data_path, output_dir, test_size=0.2, random_state=42):
    """
    Train and save the ticket categorization and priority models.

    Args:
        data_path: Path to the CSV data file
        output_dir: Directory to save the trained models
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility

    Returns:
        dict: Dictionary containing performance metrics
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Check required columns
    required_columns = ['text', 'category', 'priority']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Split data
    X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
        df['text'],
        df['category'],
        df['priority'],
        test_size=test_size,
        random_state=random_state,
        stratify=df['category']
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Train categorizer
    print("\nTraining category classifier...")
    categorizer = TicketCategorizer()
    categorizer.train(X_train, y_cat_train)

    # Evaluate categorizer
    y_cat_pred = [categorizer.predict(text) for text in X_test]
    cat_accuracy = accuracy_score(y_cat_test, y_cat_pred)
    cat_f1 = f1_score(y_cat_test, y_cat_pred, average='weighted')
    cat_precision = precision_score(y_cat_test, y_cat_pred, average='weighted')
    cat_recall = recall_score(y_cat_test, y_cat_pred, average='weighted')

    print("\nCategory Classification Metrics:")
    print(f"Accuracy: {cat_accuracy:.4f}")
    print(f"F1 Score: {cat_f1:.4f}")
    print(f"Precision: {cat_precision:.4f}")
    print(f"Recall: {cat_recall:.4f}")

    print("\nDetailed Category Classification Report:")
    print(classification_report(y_cat_test, y_cat_pred))

    # Train priority assigner
    print("\nTraining priority classifier...")
    priority_assigner = PriorityAssigner()
    priority_assigner.train(X_train, y_pri_train)

    # Evaluate priority assigner
    y_pri_pred = [priority_assigner.predict(text) for text in X_test]
    pri_accuracy = accuracy_score(y_pri_test, y_pri_pred)
    pri_f1 = f1_score(y_pri_test, y_pri_pred, average='weighted')
    pri_precision = precision_score(y_pri_test, y_pri_pred, average='weighted')
    pri_recall = recall_score(y_pri_test, y_pri_pred, average='weighted')

    print("\nPriority Classification Metrics:")
    print(f"Accuracy: {pri_accuracy:.4f}")
    print(f"F1 Score: {pri_f1:.4f}")
    print(f"Precision: {pri_precision:.4f}")
    print(f"Recall: {pri_recall:.4f}")

    print("\nDetailed Priority Classification Report:")
    print(classification_report(y_pri_test, y_pri_pred))

    # Save models
    categorizer_path = os.path.join(output_dir, "categorizer_model.pkl")
    priority_path = os.path.join(output_dir, "priority_model.pkl")

    print(f"\nSaving categorizer model to {categorizer_path}")
    categorizer.save_model(categorizer_path)

    print(f"Saving priority model to {priority_path}")
    priority_assigner.save_model(priority_path)

    print("\nModel training and evaluation complete!")

    # Return metrics
    return {
        "category_classification": {
            "accuracy": cat_accuracy,
            "f1_score": cat_f1,
            "precision": cat_precision,
            "recall": cat_recall
        },
        "priority_classification": {
            "accuracy": pri_accuracy,
            "f1_score": pri_f1,
            "precision": pri_precision,
            "recall": pri_recall
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Train ticket categorization and priority models")
    parser.add_argument("--data", required=True, help="Path to the CSV data file")
    parser.add_argument("--output", default="models", help="Directory to save the trained models")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion of the dataset to include in the test split")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    train_models(args.data, args.output, args.test_size, args.random_state)


if __name__ == "__main__":
    main()
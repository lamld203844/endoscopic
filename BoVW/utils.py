import os
import numpy as np
import pandas as pd

from tqdm import tqdm
import joblib

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from tpot import TPOTClassifier

def evaluate_model(
    X, y,
    model
) -> dict:
    """Cross validation given model"""
    # Set up Stratified K-Fold cross-validation
    kf = StratifiedKFold(n_splits=5)

    # Lists to store metric results
    precision_macro_scores = []
    precision_micro_scores = []
    recall_macro_scores = []
    recall_micro_scores = []
    f1_macro_scores = []
    f1_micro_scores = []
    balanced_accuracy_scores = []

    # Cross-validation loop
    for train_index, test_index in tqdm(kf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # # Preprocessing - hyperparams
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)

        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test fold
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        precision_macro = precision_score(y_test, y_pred, average='macro')
        precision_micro = precision_score(y_test, y_pred, average='micro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        recall_micro = recall_score(y_test, y_pred, average='micro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

        # Store metrics for this fold
        precision_macro_scores.append(precision_macro)
        precision_micro_scores.append(precision_micro)
        recall_macro_scores.append(recall_macro)
        recall_micro_scores.append(recall_micro)
        f1_macro_scores.append(f1_macro)
        f1_micro_scores.append(f1_micro)
        balanced_accuracy_scores.append(balanced_accuracy)

        # # Print classification report for each fold
        # print(f"Classification Report for Fold {len(precision_macro_scores)}:")
        # print(classification_report(y_test, y_pred))
        # print("\n")
        
    # Calculate and print the average metrics across all folds
    print("Average Metrics Across All Folds:")
    print('\n', "="*50, '\n')
    print("Precision (Macro):", np.mean(precision_macro_scores))
    print("Recall (Macro):", np.mean(recall_macro_scores))
    print("F1-Score (Macro):", np.mean(f1_macro_scores))
    print('\n', "="*50, '\n')
    print("Precision (Micro):", np.mean(precision_micro_scores))
    print("Recall (Micro):", np.mean(recall_micro_scores))
    print("F1-Score (Micro):", np.mean(f1_micro_scores))
    print('\n', "="*50, '\n')
    print("Balanced Accuracy:", np.mean(balanced_accuracy_scores))
    
    metrics = {
        "mean_precision_macro": np.mean(precision_macro_scores),
        "mean_precision_micro":  np.mean(precision_micro_scores),
        "mean_recall_macro": np.mean(recall_macro_scores),
        "mean_recall_micro": np.mean(recall_micro_scores),
        "mean_f1_macro": np.mean(f1_macro_scores),
        "mean_f1_micro": np.mean(f1_micro_scores),
    }

    return metrics


## -------- AutoML --------------------------
def TPOT_autoML(X, y, saving_path: str, metadata: str) -> Pipeline | None:
    # ------ AUTO ML start here -----
    """Run AutoML using TPOT to find the best model."""
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Initialize TPOT
    tpot = TPOTClassifier(
        generations=5,              # Number of generations to evolve
        population_size=20,         # Population size
        cv=5,                       # Cross-validation folds
        scoring="f1_macro",
        verbosity=2,                # Verbosity level
        n_jobs=-1
    )

    # Fit TPOT
    print("Running TPOT AutoML...")
    tpot.fit(X_train, y_train)

    # Export the best model pipeline
    print("Exporting the best model...")
    best_pipeline = f"{saving_path}/best_model/bestModel_for-{metadata}"
    tpot.export(f"{best_pipeline}.py")


    # Save the trained model
    best_model = tpot.fitted_pipeline_
    joblib.dump(best_model, f"{best_pipeline}.pkl")
    print(f"Best model saved as 'best_model.pkl'")

    return best_model
    # ------ AUTO ML end here -----
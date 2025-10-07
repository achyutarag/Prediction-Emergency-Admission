"""
Model Training Module for Healthcare ML Prediction
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

class HealthcareMLTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def prepare_data(self, X, y, test_size=0.2):
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=self.random_state)

    def train_logistic_regression(self, X_train, y_train, max_iter=1000):
        model = LogisticRegression(max_iter=max_iter, random_state=self.random_state)
        model.fit(X_train, y_train)
        self.models["logistic_regression"] = model
        return model

    def train_random_forest(self, X_train, y_train, n_estimators=100):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=self.random_state)
        model.fit(X_train, y_train)
        self.models["random_forest"] = model
        return model

    def evaluate_model(self, model, X_test, y_test, name):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        res = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "y_pred": y_pred, "y_proba": y_proba
        }
        self.results[name] = res
        return res

    def cross_validate(self, model, X, y, folds=5):
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        return {"accuracy_mean": acc.mean(), "accuracy_std": acc.std(),
                "auc_mean": auc.mean(), "auc_std": auc.std()}

    def plot_confusion_matrix(self, cm, title="Confusion Matrix"):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not Admitted", "Admitted"],
                    yticklabels=["Not Admitted", "Admitted"])
        plt.title(title); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.show()

    def plot_roc(self, y_test, y_proba, label="Model"):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.2f})")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend(); plt.tight_layout(); plt.show()
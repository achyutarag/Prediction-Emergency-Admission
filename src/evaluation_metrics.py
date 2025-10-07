"""
Evaluation Metrics Module for Healthcare ML Prediction
"""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

class HealthcareEvaluator:
    def bce_loss(self, y_true, y_proba):
        p = np.clip(y_proba, 1e-15, 1-1e-15)
        return -np.mean(y_true*np.log(p) + (1-y_true)*np.log(1-p))

    def evaluate(self, y_true, y_pred, y_proba=None):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        roc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
        bce = self.bce_loss(y_true, y_proba) if y_proba is not None else None
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn+fp) else 0.0
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "roc_auc": roc, "bce_loss": bce, "confusion_matrix": cm,
                "specificity": specificity}
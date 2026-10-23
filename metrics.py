"""Evaluation metrics: classification, regression, and classification derived from regression."""
import numpy as np
from sklearn.metrics import (roc_auc_score, roc_curve, cohen_kappa_score,
                             matthews_corrcoef, brier_score_loss,
                             confusion_matrix as _cm,
                             mean_squared_error, mean_absolute_error, r2_score)


def compute_metrics(y_true, y_scores, threshold=0.5, allow_brier=None):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)
    y_pred = (y_scores >= threshold).astype(int)

    tn, fp, fn, tp = _cm(y_true, y_pred, labels=[0, 1]).ravel()
    total = len(y_true)

    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fpr = fp / (tn + fp) if (tn + fp) else 0
    fnr = fn / (tp + fn) if (tp + fn) else 0
    balanced_accuracy = (recall + specificity) / 2

    mcc = float(matthews_corrcoef(y_true, y_pred)) if total else 0.0
    kappa = float(cohen_kappa_score(y_true, y_pred)) if total else 0.0

    if allow_brier is None:
        allow_brier = bool(np.all(y_scores >= 0) and np.all(y_scores <= 1))
    if allow_brier and len(set(y_true.tolist())) > 1:
        brier = float(brier_score_loss(y_true, np.clip(y_scores, 0, 1)))
    else:
        brier = None

    two_classes = len(set(y_true.tolist())) > 1
    if two_classes:
        auc = float(roc_auc_score(y_true, y_scores))
        fpr_c, tpr_c, _ = roc_curve(y_true, y_scores)
    else:
        auc = 0.0
        fpr_c, tpr_c = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    step = max(1, len(fpr_c) // 300)
    roc_curve_pts = [{'fpr': float(a), 'tpr': float(b)}
                     for i, (a, b) in enumerate(zip(fpr_c, tpr_c))
                     if i % step == 0 or i == len(fpr_c) - 1]

    return {
        'accuracy': float(accuracy), 'precision': float(precision), 'recall': float(recall),
        'specificity': float(specificity), 'f1': float(f1), 'fpr': float(fpr), 'fnr': float(fnr),
        'rocAuc': auc, 'balancedAccuracy': float(balanced_accuracy),
        'mcc': mcc, 'kappa': kappa, 'brier': brier,
        'confusionMatrix': {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)},
        'rocCurve': roc_curve_pts,
    }


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mse = float(np.mean(err ** 2)) if len(y_true) else 0.0
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err))) if len(y_true) else 0.0
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) if len(y_true) else 0.0
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true))) if len(y_true) else 0.0
    return {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
        'directionalAccuracy': dir_acc,
    }
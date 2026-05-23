"""Evaluation, metrics, and the alerting / lead-time logic
(Sections 4.1.6, 4.2, 6.3-6.8).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve,
)

from configs.config import Config, AlertConfig


# --------------------------------------------------------------------------- #
# Classification metrics
# --------------------------------------------------------------------------- #
def classification_report(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str]
) -> Dict:
    y_pred = y_proba.argmax(1)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # One-vs-rest AUC per class.
    aucs = []
    for c in range(len(class_names)):
        try:
            aucs.append(roc_auc_score((y_true == c).astype(int), y_proba[:, c]))
        except ValueError:
            aucs.append(float("nan"))

    return {
        "accuracy": acc,
        "per_class": {
            class_names[i]: {
                "precision": prec[i], "recall": rec[i],
                "f1": f1[i], "support": int(sup[i]), "auc": aucs[i],
            }
            for i in range(len(class_names))
        },
        "macro": {
            "precision": macro_p, "recall": macro_r,
            "f1": macro_f1, "auc": float(np.nanmean(aucs)),
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=range(len(class_names))
        ),
    }


def print_report(report: Dict, class_names: List[str]) -> None:
    print(f"\nOverall accuracy: {report['accuracy'] * 100:.2f}%")
    print(f"{'Class':<22}{'Prec':>7}{'Rec':>7}{'F1':>7}{'AUC':>7}{'Sup':>7}")
    print("-" * 57)
    for name in class_names:
        m = report["per_class"][name]
        print(f"{name:<22}{m['precision']:>7.3f}{m['recall']:>7.3f}"
              f"{m['f1']:>7.3f}{m['auc']:>7.3f}{m['support']:>7d}")
    macro = report["macro"]
    print("-" * 57)
    print(f"{'Macro Avg.':<22}{macro['precision']:>7.3f}{macro['recall']:>7.3f}"
          f"{macro['f1']:>7.3f}{macro['auc']:>7.3f}")


def roc_curves(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> Dict:
    """One-vs-rest ROC curve coordinates and AUC per class."""
    out = {}
    for c in range(n_classes):
        binary = (y_true == c).astype(int)
        try:
            fpr, tpr, _ = roc_curve(binary, y_proba[:, c])
            out[c] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc_score(binary, y_proba[:, c])}
        except ValueError:
            out[c] = {"fpr": np.array([0, 1]), "tpr": np.array([0, 1]), "auc": float("nan")}
    return out


# --------------------------------------------------------------------------- #
# Alerting logic (Section 4.1.6)
# --------------------------------------------------------------------------- #
@dataclass
class Alert:
    triggered: bool
    severity: str          # "None" | "Advisory" | "Warning" | "Critical"
    fault_type: int
    p_fault: float


def severity_tier(p_fault: float, cfg: AlertConfig) -> str:
    if p_fault >= cfg.critical_min:
        return "Critical"
    if cfg.warning_range[0] <= p_fault < cfg.warning_range[1]:
        return "Warning"
    if cfg.advisory_range[0] <= p_fault < cfg.advisory_range[1]:
        return "Advisory"
    return "None"


class FaultAlerter:
    """Stateful alerter requiring N consecutive windows above threshold."""

    def __init__(self, cfg: Config):
        self.cfg = cfg.alert
        self._streak = 0

    def update(self, proba_vector: np.ndarray) -> Alert:
        """Feed one window's softmax vector; return the current alert state."""
        p_normal = float(proba_vector[0])
        p_fault = 1.0 - p_normal                         # Sec. 4.1.6
        fault_type = int(np.argmax(proba_vector[1:]) + 1)

        if p_fault >= self.cfg.decision_threshold:
            self._streak += 1
        else:
            self._streak = 0

        triggered = self._streak >= self.cfg.realtime_consecutive_windows
        sev = severity_tier(p_fault, self.cfg) if triggered else "None"
        return Alert(triggered, sev, fault_type, p_fault)


# --------------------------------------------------------------------------- #
# Lead-time analysis (Section 4.2, 6.8)
# --------------------------------------------------------------------------- #
def compute_lead_time(
    fault_proba_series: np.ndarray,
    timestamps_hours: np.ndarray,
    failure_time_hours: float,
    cfg: Config,
) -> float:
    """Lead time in DAYS = T_failure - T_first_detection (Sec. 4.2).

    A detection is confirmed once P_fault stays above tau for at least
    ``leadtime_sustained_hours`` of contiguous inference.
    """
    tau = cfg.alert.decision_threshold
    sustained = cfg.alert.leadtime_sustained_hours
    above = fault_proba_series >= tau

    streak_start = None
    for i, (flag, t) in enumerate(zip(above, timestamps_hours)):
        if flag:
            if streak_start is None:
                streak_start = t
            if t - streak_start >= sustained:
                return (failure_time_hours - streak_start) / 24.0
        else:
            streak_start = None
    return 0.0

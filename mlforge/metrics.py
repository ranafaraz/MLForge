"""Scoring metrics."""

from __future__ import annotations

import numpy as np


def accuracy_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))

# Modified version of AP score from sklearn, to account for ground_truth = 0.

from functools import partial
import numpy as np

from sklearn.utils.multiclass import type_of_target
from sklearn.metrics.base import _average_binary_score
from sklearn.metrics.ranking import _binary_clf_curve


def precision_recall_curve_modified(y_true, probas_pred, pos_label=None,
                                    sample_weight=None):
    fps, tps, thresholds = _binary_clf_curve(y_true, probas_pred,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight)

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = np.ones(tps.size) if tps[-1] == 0 else tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def average_precision_score(y_true, y_score, average="macro", pos_label=1,
                            sample_weight=None):
    def _binary_uninterpolated_average_precision(
            y_true, y_score, pos_label=1, sample_weight=None):
        precision, recall, _ = precision_recall_curve_modified(
            y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

    y_type = type_of_target(y_true)
    if y_type == "multilabel-indicator" and pos_label != 1:
        raise ValueError("Parameter pos_label is fixed to 1 for "
                         "multilabel-indicator y_true. Do not set "
                         "pos_label or set pos_label to 1.")
    elif y_type == "binary":
        present_labels = np.unique(y_true)
        if len(present_labels) == 2 and pos_label not in present_labels:
            raise ValueError("pos_label=%r is invalid. Set it to a label in "
                             "y_true." % pos_label)
    average_precision = partial(_binary_uninterpolated_average_precision,
                                pos_label=pos_label)
    return _average_binary_score(average_precision, y_true, y_score,
                                 average, sample_weight=sample_weight)

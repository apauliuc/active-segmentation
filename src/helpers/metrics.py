import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric

EPS = 1e-5


def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])


def create_confusion_matrix(true, prediction, num_classes=2):
    mask = (true >= 0) & (true < num_classes)
    confusion_matrix = torch.bincount(
        num_classes * true[mask] + prediction[mask],
        minlength=num_classes ** 2
    ).reshape((num_classes, num_classes)).float()
    return confusion_matrix


def total_pixel_accuracy(cf_matrix):
    correct = torch.diag(cf_matrix).sum()
    total = cf_matrix.sum()
    accuracy = correct / (total + EPS)
    return accuracy


def per_class_pixel_accuracy(cf_matrix):
    correct_per_class = torch.diag(cf_matrix)
    total_per_class = cf_matrix.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class = nanmean(per_class_acc)
    return avg_per_class


def iou_score(cf_matrix):
    intersection = torch.diag(cf_matrix)
    iou = intersection / (cf_matrix.sum(dim=1) + cf_matrix.sum(dim=0) - intersection + EPS)
    return nanmean(iou)


def f1_score(cf_matrix):
    intersection = torch.diag(cf_matrix)
    f1 = (2. * intersection) / (cf_matrix.sum(dim=1) + cf_matrix.sum(dim=0) + EPS)
    return nanmean(f1)


def evaluation_metrics_per_batch(prediction, true, num_classes=2):
    """"
    Compute segmentation metrics for 2D images

    :param prediction: predicted classes of shape [B, 1, H, W]
    :param true: ground truth tensor of shape [B, 1, H, W]
    :param num_classes: number of classes in segmentation
    :returns tuple of scores
    """
    confusion_matrix = torch.zeros((num_classes, num_classes))
    for t, p in zip(true, prediction):
        confusion_matrix += create_confusion_matrix(t.flatten(), p.flatten(), num_classes)
    overall_pixel_accuracy = total_pixel_accuracy(confusion_matrix)
    avg_class_pixel_accuracy = per_class_pixel_accuracy(confusion_matrix)
    mean_iou = iou_score(confusion_matrix)
    mean_f1 = f1_score(confusion_matrix)

    return overall_pixel_accuracy, avg_class_pixel_accuracy, mean_iou, mean_f1


# noinspection PyAttributeOutsideInit
class SegmentationMetrics(Metric):

    def __init__(self, num_classes=2):
        self._num_classes = 2 if num_classes == 1 else num_classes
        super(SegmentationMetrics, self).__init__()

    def reset(self):
        self._confusion_matrix = torch.zeros((self._num_classes, self._num_classes))

    def update(self, output):
        if len(output) == 2:
            y_pred, y = output
        else:
            y_pred, y, kwargs = output

        y = y.cpu().type(torch.LongTensor)
        if self._num_classes == 2:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = y_pred.argmax(dim=1)

        y_pred = y_pred.cpu().type(torch.LongTensor)

        for t, p in zip(y, y_pred):
            self._confusion_matrix += create_confusion_matrix(t.flatten(), p.flatten(), self._num_classes)

    def compute(self):
        if self._confusion_matrix.sum() == 0:
            raise NotComputableError('Metrics must have at least one example before being computed')

        metrics_dict = {
            'avg_total_acc': total_pixel_accuracy(self._confusion_matrix),
            'avg_class_acc': per_class_pixel_accuracy(self._confusion_matrix),
            'avg_iou': iou_score(self._confusion_matrix),
            'avg_f1': f1_score(self._confusion_matrix)
        }
        return metrics_dict
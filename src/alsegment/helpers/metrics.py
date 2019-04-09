import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric

EPS = 1e-5


def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])


def _cf_matrix(true, prediction, num_classes=2):
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
    A = cf_matrix.sum(dim=1)
    B = cf_matrix.sum(dim=0)
    iou = intersection / (A + B - intersection + EPS)
    return nanmean(iou)


def f1_score(cf_matrix):
    intersection = torch.diag(cf_matrix)
    A = cf_matrix.sum(dim=1)
    B = cf_matrix.sum(dim=0)
    f1 = (2 * intersection) / (A + B + EPS)
    return nanmean(f1)


def evaluation_metrics(prediction, true, num_classes=2):
    """"
    Compute segmentation metrics for 2D images

    :param prediction: predicted classes of shape [B, 1, H, W]
    :param true: ground truth tensor of shape [B, 1, H, W]
    :param num_classes: number of classes in segmentation
    :returns tuple of scores
    """
    confusion_matrix = torch.zeros((num_classes, num_classes))
    for t, p in zip(true, prediction):
        confusion_matrix += _cf_matrix(t.flatten(), p.flatten(), num_classes)
    overall_pixel_accuracy = total_pixel_accuracy(confusion_matrix)
    avg_class_pixel_accuracy = per_class_pixel_accuracy(confusion_matrix)
    mean_iou = iou_score(confusion_matrix)
    mean_f1 = f1_score(confusion_matrix)

    return overall_pixel_accuracy, avg_class_pixel_accuracy, mean_iou, mean_f1


# noinspection PyAttributeOutsideInit
class SegmentationMetrics(Metric):

    def __init__(self, num_classes=lambda x: x.shape[1] + 1):
        super(SegmentationMetrics, self).__init__()
        self._num_classes = num_classes

    def reset(self):
        self._sum_total_acc = 0
        self._sum_avg_class_acc = 0
        self._sum_iou = 0
        self._sum_f1 = 0
        self._num_examples = 0

    def update(self, output):
        if len(output) == 2:
            y_pred, y = output
        else:
            y_pred, y, kwargs = output

        y = y.cpu().type(torch.LongTensor)
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.cpu().type(torch.LongTensor)

        total_acc, avg_class_acc, iou, f1 = evaluation_metrics(y_pred, y, self._num_classes(y_pred))

        if len(total_acc.shape) != 0 or len(avg_class_acc.shape) != 0 or len(iou.shape) != 0 or len(f1.shape) != 0:
            raise ValueError('evaluation_metrics did not return the average loss')

        n = self._num_classes(y)
        self._sum_total_acc += total_acc.item() * n
        self._sum_avg_class_acc += avg_class_acc.item() * n
        self._sum_iou += iou.item() * n
        self._sum_f1 += f1.item() * n
        self._num_examples += n

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Metrics must have at least one example before being computed')
        metrics_dict = {
            'avg_total_acc': self._sum_total_acc / self._num_examples,
            'avg_class_acc': self._sum_avg_class_acc / self._num_examples,
            'avg_iou': self._sum_iou / self._num_examples,
            'avg_f1': self._sum_f1 / self._num_examples
        }
        return metrics_dict

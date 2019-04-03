import torch

from alsegment.helpers.metrics import evaluation_metrics
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


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

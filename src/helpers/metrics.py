import numpy as npimport torchfrom torch import nnfrom ignite.exceptions import NotComputableErrorfrom ignite.metrics.metric import Metricfrom helpers.utils import binarize_tensorfrom losses.vae_criterion import VAECriterionfrom helpers.average_precision_mine import average_precision_score as ap_scoreEPS = 1e-5def nanmean(x):    """Computes the arithmetic mean ignoring any NaNs."""    return torch.mean(x[x == x])def create_confusion_matrix(true, prediction, num_classes=2):    mask = (true >= 0) & (true < num_classes)    confusion_matrix = torch.bincount(        num_classes * true[mask] + prediction[mask],        minlength=num_classes ** 2    ).reshape((num_classes, num_classes)).float()    return confusion_matrixdef total_pixel_accuracy(cf_matrix):    correct = torch.diag(cf_matrix).sum()    total = cf_matrix.sum()    accuracy = correct / (total + EPS)    return accuracydef per_class_pixel_accuracy(cf_matrix):    correct_per_class = torch.diag(cf_matrix)    total_per_class = cf_matrix.sum(dim=1)    per_class_acc = correct_per_class / (total_per_class + EPS)    avg_per_class = nanmean(per_class_acc)    return avg_per_classdef iou_score(cf_matrix):    intersection = torch.diag(cf_matrix)    iou = intersection / (cf_matrix.sum(dim=1) + cf_matrix.sum(dim=0) - intersection + EPS)    return nanmean(iou)def f1_score(cf_matrix):    intersection = torch.diag(cf_matrix)    f1 = (2. * intersection) / (cf_matrix.sum(dim=1) + cf_matrix.sum(dim=0) + EPS)    return nanmean(f1)def evaluation_metrics_per_batch(true, prediction, num_classes=2):    """"    Compute segmentation metrics for 2D images    :param true: ground truth tensor of shape [B, 1, H, W]    :param prediction: predicted classes of shape [B, 1, H, W]    :param num_classes: number of classes in segmentation    :returns tuple of scores    """    confusion_matrix = torch.zeros((num_classes, num_classes))    for t, p in zip(true, prediction):        confusion_matrix += create_confusion_matrix(t.flatten(), p.flatten(), num_classes)    overall_pixel_accuracy = total_pixel_accuracy(confusion_matrix)    avg_class_pixel_accuracy = per_class_pixel_accuracy(confusion_matrix)    mean_iou = iou_score(confusion_matrix)    mean_f1 = f1_score(confusion_matrix)    return overall_pixel_accuracy, avg_class_pixel_accuracy, mean_iou, mean_f1# noinspection PyAttributeOutsideInitclass SegmentationMetrics(Metric):    def __init__(self, num_classes=2, threshold=0.5, eval_ensemble=False):        self._num_classes = 2 if num_classes == 1 else num_classes        self._thres = threshold        self._eval_ensemble = eval_ensemble        super(SegmentationMetrics, self).__init__()    def reset(self):        self._confusion_matrix = torch.zeros((self._num_classes, self._num_classes))        self._ap_scores = list()    def update(self, output, process=True):        if len(output) == 2:            y_pred, y = output        else:            y_pred, y, *kwargs = output        if process and not self._eval_ensemble:            y_pred = torch.sigmoid(y_pred)        y = y.cpu().type(torch.LongTensor)        y_pred = y_pred.cpu()        for g_truth, prediction in zip(y.cpu(), y_pred.cpu()):            self._ap_scores.append(ap_score(g_truth.flatten(), prediction.flatten(), average=None, pos_label=1))            if self._num_classes == 2:                prediction = binarize_tensor(prediction, self._thres)            else:                prediction = prediction.argmax(dim=1)            prediction = prediction.type(torch.LongTensor)            self._confusion_matrix += create_confusion_matrix(g_truth.flatten(), prediction.flatten(), self._num_classes)    def compute(self):        if self._confusion_matrix.sum() == 0:            raise NotComputableError('Metrics must have at least one example before being computed')        metrics_dict = {            'avg_acc': total_pixel_accuracy(self._confusion_matrix).item(),            'avg_class_acc': per_class_pixel_accuracy(self._confusion_matrix).item(),            'avg_iou': iou_score(self._confusion_matrix).item(),            'avg_f1': f1_score(self._confusion_matrix).item(),            'mAP': np.mean(self._ap_scores)        }        return metrics_dictclass VAEMetrics(Metric):    def __init__(self, loss_fn=nn.CrossEntropyLoss(reduction='mean'),  mse_factor=0, kld_factor=0):        self.vae_criterion = VAECriterion(loss_fn, mse_factor=mse_factor, kld_factor=kld_factor)        self.total_loss = 0        self.cross_ent = 0        self.mse = 0        self.kl_div = 0        self.num_examples = 0        super(VAEMetrics, self).__init__()    def update_kld_factor(self, new_kld_factor):        self.vae_criterion.kld_factor = new_kld_factor    def update_mse_factor(self, new_mse_factor):        self.vae_criterion.mse_factor = new_mse_factor    def reset(self):        self.total_loss = 0        self.cross_ent = 0        self.mse = 0        self.kl_div = 0        self.num_examples = 0    def update(self, output):        pred, y, recon, x, mu, var = output        loss, ce, mse, kl_div = self.vae_criterion(pred, y, recon, x, mu, var)        n = pred.shape[0]        self.total_loss += loss * n        self.cross_ent += ce.item() * n        self.mse += mse.item() * n        self.kl_div += kl_div.item() * n        self.num_examples += n    def compute(self):        if self.num_examples == 0:            raise NotComputableError(                'Loss must have at least one example before it can be computed')        metrics_dict = {            'total_loss': self.total_loss / self.num_examples,            'segment_loss': self.cross_ent / self.num_examples,            'recon_loss': self.mse / self.num_examples,            'kl_div': self.kl_div / self.num_examples        }        return metrics_dictclass AverageMeter(object):    sum: float    avg: float    count: float    def __init__(self):        self.reset()    def reset(self):        self.count = 0        self.sum = 0        self.avg = 0    def update(self, value, n=1):        self.count += n        self.sum += value * n        self.avg = self.sum / self.count    @property    def average(self):        return self.avg
import torch

EPS = 1e-10


def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])


def _cf_matrix(true, prediction, num_classes=1):
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


def evaluation_metrics(true, prediction, num_classes=1):
    """"
    Compute segmentation metrics for 2D images

    :param true: ground truth tensor of shape [B, 1, H, W]
    :param prediction: predicted classes of shape [B, 1, H, W]
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

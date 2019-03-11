import os
import sys
import logging


def create_logger(logdir, name):
    log_format = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout)

    formatter = logging.Formatter(log_format)
    logger = logging.getLogger(name)
    handler = logging.FileHandler(os.path.join(logdir, "LOGGING_FILE.log"))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)

    return logger


def dice_coefficient(prediction, target):
    smooth = 1.
    num_classes = prediction.size(0)
    m1 = prediction.view(num_classes, -1)
    m2 = target.view(num_classes, -1)
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

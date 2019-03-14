import os
import sys
import logging


def setup_logger(logdir, name):
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
    batch_size = prediction.size(0)
    m1 = prediction.view(batch_size, -1)
    m2 = target.view(batch_size, -1)
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def timer_to_str(value):
    m, s = divmod(value, 60)
    h, m = divmod(m, 60)
    return "%02i:%02i:%02i" % (h, m, s)

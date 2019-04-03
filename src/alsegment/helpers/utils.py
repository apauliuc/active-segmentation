import os
import sys
import logging


def setup_logger(logdir):
    log_format = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout)

    formatter = logging.Formatter(log_format)
    logger = logging.getLogger('alsegment')
    handler = logging.FileHandler(os.path.join(logdir, "LOGGING_FILE.log"))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)

    return logger, handler


def dice_coefficient(prediction, target) -> float:
    smooth = 1e-5
    batch_size = prediction.shape[0]
    m1 = prediction.view(batch_size, -1)
    m2 = target.view(batch_size, -1)
    score_per_item = (2. * (m1 * m2).sum(dim=1) + smooth) / \
                     (m1.sum(dim=1) + m2.sum(dim=1) + smooth)
    return score_per_item.mean()


def timer_to_str(value):
    m, s = divmod(value, 60)
    h, m = divmod(m, 60)
    return "%02i:%02i:%02i" % (h, m, s)

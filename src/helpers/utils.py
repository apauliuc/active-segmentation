import os
import sys
import inspect
import logging
from typing import List


def setup_logger(logdir):
    log_format = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout)

    formatter = logging.Formatter(log_format)
    logger = logging.getLogger('scripts')
    handler = logging.FileHandler(os.path.join(logdir, "LOGGING_FILE.log"))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)

    return logger, handler


def timer_to_str(value):
    m, s = divmod(value, 60)
    h, m = divmod(m, 60)
    return "%02i:%02i:%02i" % (h, m, s)


def retrieve_class_init_parameters(class_instance) -> List:
    class_signature = inspect.signature(class_instance.__init__)
    return list(class_signature.parameters)


def binarize_tensor(x, threshold=0.5):
    cond = (x > threshold).float()
    return (cond * 1) + ((1-cond) * 0)

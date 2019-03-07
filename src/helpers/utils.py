import os
import sys
import logging


def create_logger(logdir, name):
    log_format = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout)

    # formatter = logging.Formatter(log_format)
    logger = logging.getLogger(name)
    # handler = logging.FileHandler(os.path.join(logdir, "LOGGING_FILE.log"))
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)
    # logger.setLevel(logging.INFO)

    return logger

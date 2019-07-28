"""Various logging modules."""
import logging

import coloredlogs


def get_logger():
    """Initialize Python logger that outputs to file and console."""
    logger = logging.getLogger("asdf")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("run.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = coloredlogs.ColoredFormatter(
        "%(asctime)s | %(filename)12s | %(levelname)8s | %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Fix TensorFlow doubling logs
    # https://stackoverflow.com/questions/33662648/tensorflow-causes-logging-messages-to-double
    logger.propagate = False

    return logger

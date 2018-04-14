import logging
from logging.handlers import RotatingFileHandler


def dual_logger(path_log, name="name_logger"):
    my_logger = logging.getLogger(name)
    my_logger.setLevel(logging.INFO)

    # formatting data
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')

    # first handler for appending in a log file
    file_handler = RotatingFileHandler(path_log, 'a', 1000000, 1)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    my_logger.addHandler(file_handler)

    # second handler for printing
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    my_logger.addHandler(stream_handler)

    return my_logger

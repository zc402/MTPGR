import logging

def _setup_logger():
    logger = logging.getLogger("mtpgr")
    logger.setLevel(logging.DEBUG)
    c_handler = logging.StreamHandler()
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)
    logger.debug("Logger initialized.")

    return logger

log: logging.Logger = _setup_logger()


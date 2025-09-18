import logging

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(
    name: str,
    *,
    level: int | str = logging.DEBUG,
    format: str = "[%(asctime)s.%(msecs)d][%(levelname)s] %(message)s",
    datefmt: str = "%H:%M:%S",
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(format, datefmt=datefmt))
        logger.addHandler(handler)
    return logger

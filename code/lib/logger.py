#!/usr/bin/env python
import sys
import logging


class Logger:

    """Singleton logger class"""

    __instance = None
    __logger = None

    @staticmethod
    def get_logger():
        if Logger.__instance is None:
            Logger()
        return Logger.__logger

    def __init__(self):
        """Initialize logger instance"""

        if Logger.__instance is not None:
            raise Exception("This class is singleton!")

        Logger.__instance = self

        root = logging.getLogger()
        root.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)

        # assign logger to the logging singleton
        Logger.__logger = root

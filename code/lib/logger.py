#!/usr/bin/env python
import logging
import sys

__all__ = ["logger"]

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# handler
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)

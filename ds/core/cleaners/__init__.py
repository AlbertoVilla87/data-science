"""
Cleaners package.
"""
import logging
import os

from ds.core.utils.logger import DsLogger

# Logger
logging.setLoggerClass(DsLogger)

logger = logging.getLogger("cleaners")
logger.setLevel(logging.DEBUG)

logger.propagate = False

CONF_INI = "ds/core/cleaners/scripts/local.ini"
"""
TALKBOT package.
"""
import logging
import os

from ds.core.utils.logger import DsLogger

# Logger
logging.setLoggerClass(DsLogger)

logger = logging.getLogger("myberts")
logger.setLevel(logging.DEBUG)

logger.propagate = False

CONF_INI = "ds/sol/myberts/scripts/local.ini"
# -*- coding: utf-8 -*-
"""
Process labeled data
"""
import logging
import os

from configparser import ConfigParser
from datetime import datetime

from ds.sol.talkbot import logger, CONF_INI
from ds.sol.talkbot.processing import text_proc

def _main():
    try:
        cfg = ConfigParser()
        cfg.read(CONF_INI)

        # Log conf
        log_file = os.path.join(cfg["LOGS"]["Path"],
                                datetime.now().strftime('proclabel_%Y_%m_%d_%H_%M_%S.log'))

        logger.set_file_logs(level=logging.INFO, filename=log_file)
        data_labeled_path = cfg["INPUTS"]["data_labeled"]

        logger.info("Process labeled data...")
        text_proc.n3_to_csv(data_labeled_path)
        logger.info("Process labeled data done")

    except Exception:  # pylint: disable=broad-except
        logger.exception("Process failed")


if __name__ == "__main__":
    _main()

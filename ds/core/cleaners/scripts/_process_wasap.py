# -*- coding: utf-8 -*-
"""
Process wasap
"""
import logging
import os

from configparser import ConfigParser
from datetime import datetime

from ds.core.cleaners import logger, CONF_INI, whatsapp

def _main():
    try:
        cfg = ConfigParser()
        cfg.read(CONF_INI)
        print(cfg.items())

        # Log conf
        log_file = os.path.join(cfg["LOGS"]["Path"],
                                datetime.now().strftime('wasap_%Y_%m_%d_%H_%M_%S.log'))

        logger.set_file_logs(level=logging.INFO, filename=log_file)
        data_path = cfg["INPUTS"]["whatsapp"]

        logger.info("Process wasap...")
        whatsapp.to_csv(data_path)
        logger.info("Process wasap done")

    except Exception:  # pylint: disable=broad-except
        logger.exception("Process failed")


if __name__ == "__main__":
    _main()
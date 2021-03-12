# -*- coding: utf-8 -*-
"""
BERT fit using SAB dataset
"""
import logging
import os

from configparser import ConfigParser
from datetime import datetime

import pandas as pd

from ds.core.cleaners import sab
from ds.sol.myberts import logger, CONF_INI
from ds.sol.myberts.processing import process
from ds.sol.myberts.modeling import my_bert

def _main():
    try:
        cfg = ConfigParser()
        cfg.read(CONF_INI)

        # Log conf
        log_file = os.path.join(cfg["LOGS"]["Path"],
                                datetime.now().strftime('train_bert_%Y_%m_%d_%H_%M_%S.log'))

        logger.set_file_logs(level=logging.INFO, filename=log_file)
        bert_model = cfg["MODELS"]["bert_model"]
        bert_tokenizer = cfg["MODELS"]["bert_tokenizer"]
        data_labeled_path = cfg["INPUTS"]["data_train"]
        logger.info("Process labeled data...")
        data = sab.process(data_labeled_path)
        logger.info("Balance data...")
        data = process.balance_data(data, sab.POLR_ATTR)
        logger.info("Train BERT..")
        my_bert.fit(data, sab.CONT_ATTR, sab.LABL_ATTR,
                    bert_model, bert_tokenizer, sab.DICT_LABEL)
        logger.info("Analyze sentiment done")

    except Exception:  # pylint: disable=broad-except
        logger.exception("Process failed")


if __name__ == "__main__":
    _main()
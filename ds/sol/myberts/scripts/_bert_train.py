# -*- coding: utf-8 -*-
"""
BERT fit
"""
import logging
import os

from configparser import ConfigParser
from datetime import datetime

import pandas as pd

from ds.sol.myberts import logger, CONF_INI
from ds.sol.myberts.modeling import my_bert
from ds.sol.myberts.processing import text_proc
from ds.sol.myberts.utils import corpus as cp

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
        data_labeled_path = cfg["INPUTS"]["data_labeled"]
        encoding = cfg["ENCODING"]["spa"]
        logger.info("Process labeled data...")
        data = text_proc.proc_data_label(data_labeled_path, encoding, cp.DICT_LABEL)
        logger.info("Train BERT..")
        my_bert.fit(data, cp.CONT_ATTR, cp.LABL_ATTR,
                    bert_model, bert_tokenizer, cp.DICT_LABEL)
        logger.info("Analyze sentiment done")

    except Exception:  # pylint: disable=broad-except
        logger.exception("Process failed")


if __name__ == "__main__":
    _main()
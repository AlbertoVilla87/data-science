# -*- coding: utf-8 -*-
"""
BERT predict
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
                                datetime.now().strftime('pred_%Y_%m_%d_%H_%M_%S.log'))

        logger.set_file_logs(level=logging.INFO, filename=log_file)
        bert_model = cfg["MODELS"]["bert_model"]
        bert_tokenizer = cfg["MODELS"]["bert_tokenizer"]
        senti_bert = cfg["OUT_MODELS"]["model_selected"]
        data_path = cfg["OUTPUTS"]["proc"]
        encoding = cfg["ENCODING"]["spa"]
        logger.info("Process labeled data...")
        data = pd.read_csv(data_path, sep=";", encoding=encoding)
        sentences = data["text"].values
        logger.info("BERT prediction..")
        my_bert.predict(bert_model, senti_bert, bert_tokenizer, sentences, cp.DICT_LABEL)
        logger.info("BERT prediction done")

    except Exception:  # pylint: disable=broad-except
        logger.exception("Process failed")


if __name__ == "__main__":
    _main()
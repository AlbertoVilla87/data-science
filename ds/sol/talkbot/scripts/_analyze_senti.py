# -*- coding: utf-8 -*-
"""
Sentiment Analysis
"""
import logging
import os

from configparser import ConfigParser
from datetime import datetime

import pandas as pd

from ds.sol.talkbot import logger, CONF_INI
from ds.sol.talkbot.processing import text_senti
from ds.sol.talkbot.utils import corpus

def _main():
    try:
        cfg = ConfigParser()
        cfg.read(CONF_INI)

        # Log conf
        log_file = os.path.join(cfg["LOGS"]["Path"],
                                datetime.now().strftime('senti_%Y_%m_%d_%H_%M_%S.log'))

        logger.set_file_logs(level=logging.INFO, filename=log_file)
        data_path = cfg["OUTPUTS"]["proc"]
        bert_model = cfg["MODELS"]["bert_model"]
        bert_tokenizer = cfg["MODELS"]["bert_tokenizer"]
        data = pd.read_csv(data_path, sep=";")
        sentences = data[corpus.TEXT_ATTR].values
        logger.info("Analyze sentiment...")
        text_senti.tokenize_bert_sents(sentences, bert_model, bert_tokenizer)
        logger.info("Analyze sentiment done")

    except Exception:  # pylint: disable=broad-except
        logger.exception("Process failed")


if __name__ == "__main__":
    _main()
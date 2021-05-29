# -*- coding: utf-8 -*-
"""
BERT predict using SAB dataset
"""
import logging
import os

from configparser import ConfigParser
from datetime import datetime

import pandas as pd

from ds.sol.myberts import logger, CONF_INI
from ds.sol.myberts.modeling import my_bert
from ds.core.cleaners import sab, whatsapp, tweeter

# Settings
TEXT_FIELD = "text"
LABEL_FIELD = "label"
PROB_FIELD = "prob"

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
        data_path = cfg["INPUTS"]["data_predict"]
        logger.info("Process labeled data...")
        data = pd.read_csv(data_path, sep=";", encoding="utf-8")
        sentences = data[TEXT_FIELD].values
        sentences = [tweeter.clean_text(sent) for sent in sentences]
        logger.info("BERT prediction..")
        labels_pred, probs = my_bert.predict(bert_model, senti_bert, bert_tokenizer,
                        sentences, sab.DICT_LABEL)
        out_data = pd.DataFrame([labels_pred, probs]).T
        out_data.columns = [PROB_FIELD, LABEL_FIELD]
        out_data = pd.concat([data, out_data], axis=1)
        out_data.to_csv(cfg["OUTPUTS"]["data_predicted"], sep=";", index=False, encoding="utf-8")
        logger.info("BERT prediction done")

    except Exception:  # pylint: disable=broad-except
        logger.exception("Process failed")


if __name__ == "__main__":
    _main()
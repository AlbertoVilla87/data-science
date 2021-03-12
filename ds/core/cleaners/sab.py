# -*- coding: utf-8 -*-
"""
Text Processing for SAB
http://sabcorpus.linkeddata.es/
"""

import pandas as pd
from ds.core.cleaners import tweeter

# Settings

POLR_ATTR = "hasPolarity"
CONT_ATTR = "content"
NEUT_ATTR = "neutral"
LABL_ATTR = "label"
SENT_ATTR = "sentences"

CLASSES = ["negative", "neutral", "positive"]
DICT_LABEL = {"negative":0, "neutral":1, "positive":2}
ENCODING = "latin-1"

# Process data label

def process(file_path):
    """
    Process SAB Corpus (Spanish Corpus for Sentiment Analysis towards Brands)
    :param string file_path: path of the labeled sentiment file
    :param string encoding: encoding
    """
    data = pd.read_csv(file_path, sep=";", encoding=ENCODING)
    data.loc[data[POLR_ATTR].isnull(), POLR_ATTR] = NEUT_ATTR
    data[CONT_ATTR] = data[CONT_ATTR].apply(lambda x: tweeter.clean_text(x))
    data[LABL_ATTR] = data[POLR_ATTR].map(DICT_LABEL)
    return data
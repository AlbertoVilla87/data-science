# -*- coding: utf-8 -*-
"""
Text Processing
"""
import re
from configparser import ConfigParser
import pandas as pd
import numpy as np

from ds.sol.myberts import logger, CONF_INI
from ds.sol.myberts.utils import corpus as cp

CFG = ConfigParser()
CFG.read(CONF_INI)

# Wasap TXT to CV2

def wasap_to_csv(file_path):
    """
    Idenfity the wasap components and save in a csv
    :param string file_path: path of the wasap conversation
    """
    data = open(file_path, "r", encoding="utf-8").readlines()
    data = [line.replace('\n', ' ').replace('\r', '') for line in data]
    date = []
    user = []
    text = []
    num_lines = len(data)
    index = 1
    for line in data:
        logger.info("Line: " + str(index) + "/" + str(num_lines))
        if " - " in line:
            date.append(line.split(" -")[0])
            user.append(line.split(" - ")[1].split(":")[0])
            text.append(line.split(" - ")[1].split(":")[1])
        index += 1
    pro_data = pd.DataFrame([date, user, text]).T
    pro_data.columns = [cp.DATE_ATTR, cp.USER_ATTR, cp.TEXT_ATTR]
    pro_data.to_csv(CFG["OUTPUTS"]["proc"], sep=";", index=False)

# Process data label

def proc_data_label(file_path, encoding, label_dict):
    """
    Process data label to remove empty emotions oo another kind
    of processing
    :param string file_path: path of the labeled sentiment file
    """
    data = pd.read_csv(file_path, sep=";", encoding=encoding)
    data.loc[data[cp.POLR_ATTR].isnull(), cp.POLR_ATTR] = cp.NEUT_ATTR
    data[cp.CONT_ATTR] = data[cp.CONT_ATTR].apply(lambda x:
                                                  clean_text(x))
    data[cp.LABL_ATTR] = data[cp.POLR_ATTR].map(label_dict)
    return data

# Convert categorical to numeric

def convert_cat_num(label_col):
    """
    Convert and create a categorical column into numerical.
    BERT models works with numerical values
    :param iterable label_col: categorical column
    :return: numerical column
    """
    label_dict = {}
    possible_labels = np.unique(label_col)
    for index, possible_labels in enumerate(possible_labels):
        label_dict[possible_labels] = index
    return label_dict

# Clean text

def clean_text(text):
    """
    Apply cleansing techniques
    :param string text: text
    :return: cleaned text
    """
    text = re.sub(r"http\S+", "", text)
    return text

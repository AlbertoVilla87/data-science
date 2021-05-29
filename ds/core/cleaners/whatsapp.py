# -*- coding: utf-8 -*-
"""
Text Processing for whatsapp
"""
from configparser import ConfigParser
import pandas as pd

from ds.core.cleaners import logger, CONF_INI

# Wasap TXT to CV2

DATE_ATTR = "date"
USER_ATTR = "user"
TEXT_ATTR = "text"

CFG = ConfigParser()
CFG.read(CONF_INI)

ENCODING = "utf-8"

def to_csv(file_path):
    """
    Idenfity the wasap components and save in a csv
    :param string file_path: path of the wasap conversation
    :return: void
    """
    data = open(file_path, "r", encoding=ENCODING).readlines()
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
    pro_data.columns = [DATE_ATTR, USER_ATTR, TEXT_ATTR]
    pro_data.to_csv(CFG["OUTPUTS"]["whatsapp"], sep=";", index=False)
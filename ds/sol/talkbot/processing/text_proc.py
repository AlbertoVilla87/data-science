# -*- coding: utf-8 -*-
"""
Wasap Processing
"""
import pandas as pd

from ds.sol.talkbot import logger, CONF_INI

def to_csv(file_path):
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
    pro_data.columns = ["date", "user", "text"]
    print(pro_data)
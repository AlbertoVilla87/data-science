# -*- coding: utf-8 -*-
"""
Wasap Processing
"""
import pandas as pd
from configparser import ConfigParser

import rdflib

from ds.sol.talkbot import logger, CONF_INI
from ds.sol.talkbot.utils import corpus

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
    pro_data.columns = [corpus.DATE_ATTR, corpus.USER_ATTR, corpus.TEXT_ATTR]
    pro_data.to_csv(CFG["OUTPUTS"]["proc"], sep=";", index=False)

# Read corpus n3

def n3_to_csv(file_path):
    """
    Idenfity the n3 components and save in a csv
    :param string file_path: path of the labeled sentiment file
    """

    graph = rdflib.Graph()
    graph.parse(file_path, format='n3')
    for res in graph:
        print (res)
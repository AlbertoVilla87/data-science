# -*- coding: utf-8 -*-
"""
Text Processing for BERT
"""

import numpy as np

# Balance data

def balance_data(data, category):
    """
    Balance data based on a category/field
    :param pandas data: [description]
    :param string category: field to balance
    :return: data balanced
    """
    g = data.groupby(category)
    return g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
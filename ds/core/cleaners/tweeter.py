# -*- coding: utf-8 -*-
"""
Text Processing for Tweeter
"""

import re

# Clean text

def clean_text(text):
    """
    Apply cleansing techniques for tweet
    :param string text: text
    :return: cleaned text
    """
    text = text.lower()
    text = re.sub(r"http\S+", "URL", text)
    mentions = re.compile(r'((?<=\W)|^)(@\w+)(\s*@\w+)*')
    email = re.compile(r'[\w.+-]+@([a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+')
    numbers = re.compile(r'[0-9]+')
    emoji_pattern = re.compile("["
                               "\U0001F600-\U0001F64F"  # emoticons
                               "\U0001F300-\U0001F5FF"  # symbols & pictographs
                               "\U0001F680-\U0001F6FF"  # transport & map symbols
                               "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = mentions.sub('USER', text)
    text = email.sub('MAIL', text)
    text = emoji_pattern.sub(r'', text)
    text = numbers.sub(r'', text)
    text = remove_punctuation(text)
    text = re.sub(' +', ' ', text)                  # Multiple white spaces
    text = text.strip()

    return text

def remove_punctuation(text):
    """
    Remove punctuation
    :param string text: [description]
    :return: cleaned text
    """
    punctuation = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
    spaces = ' ' * len(punctuation)
    return text.translate(str.maketrans(punctuation, spaces))

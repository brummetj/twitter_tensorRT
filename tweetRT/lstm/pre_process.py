import re
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from tweetRT.lstm.constants import hyper_parameter
from tweetRT.utils.logger import Logger

logger = Logger("PreProcess")


class PreProcess:
    """
    Removes punctuation, question marks, etc,.. and only leaves alphanumeric characters
    """
    def __init__(self):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.strip_special_chars =  re.compile("[^A-Za-z0-9 ]+")

    def clean_sentences(self, string):
        string = string.lower().replace("<br />", " ")
        return re.sub(self.strip_special_chars, "", string.lower())

    def get_sentence_matrix(self, sentence, word_list):
        """
        :param sentence: a string given sentence
        :param word_list: the word vector storage.
        :return: a sentence matrix.
        """
        arr = np.zeros([hyper_parameter['BATCH_SIZE'], hyper_parameter['MAX_SEQ_LENGTH']])
        sentence_matrix = np.zeros([hyper_parameter['BATCH_SIZE'],
                                    hyper_parameter['MAX_SEQ_LENGTH']], dtype='int32')
        cleaned_sentence = self.clean_sentences(sentence)

        split = cleaned_sentence.split()

        for i, word in enumerate(split):
            try:
                sentence_matrix[0, i] = word_list.index(word)
            except ValueError:
                sentence_matrix[0, i] = 399999 # Vector for unknown words

        return sentence_matrix


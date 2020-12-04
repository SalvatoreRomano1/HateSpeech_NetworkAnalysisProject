# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:05:32 2019

@author: alberto
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections

from sklearn.feature_extraction.text import CountVectorizer

import tweepy as tw

import nltk
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

import re
import networkx as nx

import warnings
warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)
sns.set_style("whitegrid")

#%%

a = pd.read_csv("ex_1_gephi.csv")

b = a.to_numpy()

c = b[:, 1:]

c = np.int32(c)
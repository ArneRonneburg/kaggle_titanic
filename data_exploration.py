# -*- coding: utf-8 -*-
"""
Created on Sat May 15 13:56:03 2021

@author: Arne
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sklearn as sk

path=r"C:\Users\Arne\Documents\DataScience\kaggle_titanic/"

train=pd.read_csv(path+"train.csv")
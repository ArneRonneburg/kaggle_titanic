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

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.neural_network import MLPRegressor as NN

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
# itimp=IterativeImputer()
# X=itimp.fit_transform(X)

path=r"C:\Users\Arne\Documents\DataScience\kaggle_titanic/kaggle_titanic/"

train=pd.read_csv(path+"train.csv")

Xdata=train.drop(columns='Survived')
y=train.Survived
X=Xdata.drop(columns=['Name'])
categorical=[i for i in X.columns if X[i].dtype !=float and X[i].dtype!=int]
##schreib ne methode, die erst dropna, dann label encoding und na wieder rein
for c in ['Sex', 'Embarked']:
    nonan=X[c].dropna()
    
    le=LabelEncoder()

    le.fit(nonan)
    nonans=le.transform(nonan)
    X[c][nonan.index]=nonans
    
cabin_new=[]
nanlist=[]

cabin=X.Cabin.dropna()
index=cabin.index
cabins=[]
for i in range(0, len(X)):
    
    if str(X.Cabin.iloc[i])!='nan':
        cabins.append(str(X.Cabin.iloc[0])[0])
    else:
        cabins.append(np.NaN)
    

le=LabelEncoder()
le.fit(cabin)
cabin=le.transform(cabin)
X['Cabin'][index]=cabin
    
X_train, X_test, y_train, y_test=tts(X,y)



model=RFR()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
accur=1-sum(abs(y_pred-y_test))/len(y_pred)
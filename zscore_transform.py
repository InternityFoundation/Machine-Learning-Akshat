# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 23:50:35 2018

@author: admin

Standardize data
It is most suitable for techniques that assume a Gaussian distribution in the input variables and 
work better with rescaled data, such as linear regression, logistic regression and linear discriminate analysis.

Concept : v(i)'=(v(i)-mean(v))/st_deviation(v)

0<=i<=Rows
"""


import pandas as pd
import scipy
import numpy
from sklearn.preprocessing import StandardScaler
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df=pd.read_csv(url,names=names)
array = df.values
X=array[:,0:8]
Y=array[:,8]
scaler=StandardScaler()
rescaledx=scaler.fit_transform(X)
numpy.set_printoptions(precision=3)
print(rescaledx[0:5,:])


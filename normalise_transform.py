# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 00:10:17 2018

@author: admin

Normalize Data
Normalizing refers to rescaling each observation (row) to have a length of 1 (called a unit norm in linear algebra).

This preprocessing can be useful for sparse datasets (lots of zeros) with attributes of varying scales when using algorithms that
 weight input values such as neural networks and algorithms that use distance measures such as K-Nearest Neighbors.

Concept : v(i)'=v(i)/(sqrt(sum(v(j)*v(j),0<=j<=Cols)))

0<=i<=Columns

"""

import pandas as pd
import scipy
import numpy
from sklearn.preprocessing import Normalizer
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df=pd.read_csv(url,names=names)
array = df.values
X=array[:,0:8]
Y=array[:,8]
scaler=Normalizer()
rescaledx=scaler.fit_transform(X)
numpy.set_printoptions(precision=3)
print(rescaledx[0:5,:])

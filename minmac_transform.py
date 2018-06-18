# -*- coding: utf-8 -*-
"""
Spyder Editor

Rescale data

Attributes are often rescaled into the range between 0 and 1
 It is useful for algorithms that weight inputs like regression and 
 neural networks and algorithms that use distance measures like K-Nearest Neighbors.
 
This is a temporary script file.
Concept: v(i)'=((v(i)-min(v))/(max(v)-min(v)))*(new_start - new_end) + new_start

0<=i<=Rows

"""

import pandas as pd
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df=pd.read_csv(url,names=names)
array = df.values
X=array[:,0:8]
Y=array[:,8]
scaler=MinMaxScaler(feature_range=(0,1))
rescaledx=scaler.fit_transform(X)
numpy.set_printoptions(precision=3)
print(rescaledx[0:30,:])
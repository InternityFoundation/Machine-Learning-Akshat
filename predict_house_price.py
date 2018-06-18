# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 01:26:01 2018

@author: admin
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

def get_data(file_name):
    data = pd.read_csv(file_name,names=['xs','ys'])
    array=data.values
    
    x_param = array[0:,0:1]
    y_param = array[0:,1:2]
    x_parameter=[]
    y_parameter=[]
    for x in x_param:
        x_parameter.append([float(x)])
    for y in y_param:
        y_parameter.append(float(y))

    return (x_parameter),(y_parameter)

def show_linear_line(X_parameters,Y_parameters):
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    plt.scatter(X_parameters,Y_parameters,color='b', label="train data")
    predict_value=[[700.0]]
    predict_outcome = [regr.predict(predict_value)]
    plt.scatter(predict_value,predict_outcome,color='green', label="test data")
    plt.plot(X_parameters,regr.predict(X_parameters),color='red',linewidth=4)

    plt.xticks(())
    plt.yticks(())
    plt.legend(loc=2)
    plt.xlabel("house sizes")
    plt.ylabel("net worths")

    plt.savefig("test.png")
    plt.show()

x,y = get_data('input_data.csv')
"""
print (x)
print (y)
"""
def linear_model_main(X_parameters,Y_parameters,predict_value):
 
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    predict_outcome = regr.predict(predict_value)
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome
    predictions['rss']=regr.score(x,y)
    return predictions

predict_value = 700
result = linear_model_main(x,y,predict_value)
print ("Intercept value " , result['intercept'])
print ("coefficient" , result['coefficient'])
print ("Predicted value: ",result['predicted_value'])
print ("r-sq-score",result['rss'])
show_linear_line(x,y)


"""
        polynomial regression 
                                """

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(x)
X_test_poly = poly.fit_transform(x)

regressor = LinearRegression()

regressor.fit(X_train_poly,y)

y_pred = regressor.predict(X_test_poly)

plt.scatter(x,y,color='b', label="train data")

plt.plot(x,regressor.predict(X_test_poly),color='red',linewidth=4)

plt.xticks(())
plt.yticks(())
plt.legend(loc=2)

plt.xlabel("house size")
plt.ylabel("net worth")

plt.savefig("test2.png")
plt.show()

ex_var_score = explained_variance_score(y, y_pred)
m_absolute_error = mean_absolute_error(y, y_pred)
m_squared_error = mean_squared_error(y, y_pred)
r_2_score = r2_score(y, y_pred)

print("Explained Variance Score: "+str(ex_var_score))
print("Mean Absolute Error "+str(m_absolute_error))
print("Mean Squared Error "+str(m_squared_error))
print("R Squared Score "+str(r_2_score))
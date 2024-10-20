#!/usr/bin/env python
# coding: utf-8

# In[1]:


#The librarie we need to read, visualize data and run the linear regression model

import pandas as pd
import numpy as np 
import math

#Linear Model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split #splot train-test data 

#Metrics
from sklearn import metrics
from sklearn.metrics import r2_score


# In[2]:


data=pd.read_csv("C:/Users/Shamba Chakraborty/Downloads/archive (3)/Twitter Stock Market Dataset.csv")
data


# In[3]:


data.dropna(inplace=True)
data


# In[4]:


X = data[['High', 'Low', 'Open', 'Volume']].values
y = data['Close'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X,y)


# In[5]:


#Create a linear regression model

regressor = LinearRegression()

#Fit the train data into the regression model

regressor.fit(X_train, y_train)


# In[6]:


predicted = regressor.predict(X_test)


# In[7]:


dframe = pd.DataFrame({'Actual':y_test.flatten(), 'Predicted':predicted.flatten()})
dframe.head(20)


# In[8]:


print('Mean Absolute Error:' , metrics.mean_absolute_error(y_test, predicted))
print('Mean Squared Error:' , metrics.mean_squared_error(y_test, predicted))
print('Root Mean Squared Error:' , np.sqrt(metrics.mean_squared_error(y_test, predicted)))
print('R2_score:', r2_score(dframe['Actual'], dframe['Predicted']))


# In[ ]:





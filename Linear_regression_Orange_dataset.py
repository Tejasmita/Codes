#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df= pd.read_csv("https://raw.githubusercontent.com/manishanker/Statistics_ML_26Aug/master/doubts/orange_dataset.csv")


# In[4]:


df.head()


# In[6]:


df.shape


# In[7]:


df.columns


# In[9]:


df.info()


# In[10]:


df.describe()


# In[13]:


plt.scatter(df["circumference"],df["age"])


# In[19]:


from sklearn.linear_model import LinearRegression
import numpy as np
model= LinearRegression()


# In[21]:


X= df["circumference"].values.reshape(-1,1)


# In[22]:


y = df["age"]


# In[23]:


model.fit(X,y)


# In[25]:


model.score(X,y)


# In[28]:


y_pred= model.predict(X)


# In[29]:


y_pred


# In[30]:


y_true = y


# In[32]:


from sklearn.metrics import mean_squared_error


# In[33]:


mean_squared_error(y_true, y_pred)


# In[34]:


model.coef_


# In[35]:


model.intercept_


# In[55]:


def viz_linear_regression():
    plt.scatter(X, y, color='green')
    plt.plot(X, model.predict(X), color='blue')
    plt.title('Analysis of Orange dataset (Linear Regression)')
    plt.xlabel('Circumference')
    plt.ylabel('Age')
    plt.show()
    return


# In[56]:


viz_linear_regression()


# In[57]:


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)


# In[58]:


# Visualizing the Polymonial Regression results
def viz_polymonial():
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return
viz_polymonial()


# In[59]:


mean_squared_error(y_true, pol_reg.predict(poly_reg.fit_transform(X)))


# In[ ]:





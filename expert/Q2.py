#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# This notebook contains all code and discussions for the __expert iteration__ of the research question involving __age, gender and malignancy__. [Previously](../intermediate/Q2.ipynb) we focused on conditional probability by using some of the features in the descriptions files to come up with a probabilty that a patient has a malignant lesion. This notebook looks to predict a person's age using all of the other features of the image. Before we do this prediction we use a dimensionality reduction method to make sure we only use the most relevant (meaningful) features. In addition to doing prediction with a regular linear regression model we also use ridge and lasso regression.
# 
# 
# # Table of Content
# 
# 1. [Setup](#setup)
# 2. [Data Loading](#loading_cell)
# 3. [Data Preprocessing](#data_prep)
# 3. [Analysis](#analysis)
# 4. [Discussion](#discussion)

# # Setup <a id='setup'></a>
# Import the appropriate packages.<br>
# Notice that this time we are importing multiple packages from sklearn. [Sklearn](https://scikit-learn.org/stable/) is a very popular package for doing machine learning in python.

# In[39]:



# # Loading <a id='loading_cell'></a>
# Here we get all of the images from their respective folders. <br>

# In[40]:





# Here we turn the json files we have into one dataframe using pandas.

# In[59]:
print(len(dsc_filepaths))


        #image_df.loc[i] = list(j["meta"]["clinical"].values())


# In[60]:


#image_df.head()


# # Data Preprocessing <a id='data_prep'>
# Linear Regression only works if all your data is in a numeric representation. So we must turn all of text data (displayed above) into integers or floats.

# In[61]:




# There are also missing values in our data there are a few ways to deal with __missing values__: <br>
# Imputation (using some statistical method to come up with a value) <br>
# *Deleting* (removing the rows that are not complete) <br>
# Choosing a single value to replace all missing values with <br>

# In[75]:



print(len(image_df))
#print(len(image_df2))

# In addition to turning all of the text data into integers. We also turn all of the ages into floats between 0 and 1, the reason why we do this is because it keeps the range of possible values small which helps the models accuracy by avoiding a large range of possibilities.

# In[62]:





# Now that all of our data is ready to be used for our linear regression model, we need to extract our target variable (age) and our predictor variables (everything else)

# In[78]:




print(x.shape)
# One thing that can be done to make the model more efficient is reducing the amount of predictor variables we use to predict our target. A very popular method is Principal Component Analysis (PCA).

# In[80]:




print("Done with PCA")
# PCA lets us know how much of the variance in our data is explained by the columns. Here we set the threshold to 10% meaning, that if the variable explains less than 10% of the variance we discard it.

# In[81]:




# # Analysis <a id='analysis'></a>
# Now that we have our reduced data we can split it into our training and testing set. <br>
# Here we use a split of 33% for our testing set and 67% for our training set.

# In[82]:





# ### To make our predictions we will be using regular linear regression, ridge regression, and lasso regression. 
# 
# ### Ridge and Lasso are just like regular linear regression except they include penalties.
# 
# A super important fact we need to notice about Ridge regression is that it enforces the β coefficients to be lower, but it **does not enforce them to be zero**. That is, it will not get rid of irrelevant features but rather minimize their impact on the trained model. <br>
# 
# The Lasso method overcomes the disadvantage of Ridge regression by not only punishing high values of the coefficients β but actually **setting them to zero** if they are not relevant. Therefore, you might end up with fewer features included in the model than you started with, which is a huge advantage. <br>
# 
# Seeing as we already used PCA to exclude features that do not give us much information, in this case it is safe to use Ridge Regression, but results for all versions are displayed.

# In[83]:




print("Done with Training models")
# In[84]:





# MSE (Mean Squared Error) represents the **difference between the original and predicted values** which are extracted by squaring the average difference over the data set. It is a measure of **how close a fitted line is to actual data points**. The lesser the Mean Squared Error, the closer the fit is to the data set. The MSE has the units squared of whatever is plotted on the vertical axis. <br>
# 
# RMSE (Root Mean Squared Error) is the error rate by the square root of MSE. RMSE is the most easily interpreted statistic, as it has the **same units as the quantity plotted on the vertical axis or Y-axis**. RMSE can be directly interpreted in terms of measurement units, and hence it is a better **measure of fit** than a correlation coefficient.

# In[85]:





# # Discussion <a id='discussion'></a>
# Is it possible to accurately predict someone's age using one of these models?

# In[1]:


#get_ipython().system('jupyter nbconvert --to script Q2.ipynb')


# In[ ]:





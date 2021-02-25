#!/usr/bin/env python
# coding: utf-8

# # Overview 
# 
# This notebook contains all code and discussions for the __novice iteration__ of the research question involving __the distribution of different lesions__. The specific question asked, what is the distribution of the different types of lesion diagnosis? We specifically want to know if it is skewed.  At the novice level, this requires data acquisition and minimal pre-processing, coupled with exploratory charts.
#   
# 
# 
# # Table of Contents
# 
# 1. [Setup](#setup) <br>
# 2. [Data Loading](#load) <br>
# 
# 3. [Data wrangling](#dw) <br>
#     i. [Population](#pop) <br>
# 
# 4. [Visualization](#viz)
# 5. [Discussion](#Discuss)

# <a id='setup'></a>
# # Setup
# Import packages

# In[3]:





# # Load data from directories <a id='load'></a>
# Find current working directory

# Create paths to image and description folders

# In[3]:







# # Data wrangling <a id='dw'></a>

# In[5]:





# ### Populate the above lists by reading in data from the JSON files  <a id="pop"></a>
# We also have to make sure that the user gives us both 

# In[6]:



#             if len(age) > len(lesion_diagnosis):
#                 age.pop(-1)
#             else:
#                 lesion_diagnosis


# Dataframe building for easy visualization.

# In[7]:




# Here we create a csv file with the counts of all lesion diagnoses, we'll use this later :) 

# In[8]:



#diagnosis_distribution


# # Visualization <a id='viz'></a>

# In[9]:





# # What did you find out about the distribution of lesion diagnoses? <a id="Discuss"></a>
# Does it resemble a common distribution?

# In[ ]:





# In[ ]:





# In[ ]:


#get_ipython().system('jupyter nbconvert --to script Q3.ipynb')


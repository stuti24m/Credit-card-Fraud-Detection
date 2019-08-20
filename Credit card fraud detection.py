
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy


# In[3]:


data = pd.read_csv("C:\\Users\\subroto mittra\\Desktop\\STUTI\\Anaconda\\creditcard.csv")
data


# In[4]:


data.head()


# # Exploring the dataset
# 

# In[5]:


print(data.columns)


# In[6]:


# checking for null values
data.isnull().values.any()


# In[7]:


# exploring the number of rows and columns
print(data.shape)


# In[8]:


print(data.describe())


# In[9]:


data = data.sample(frac = 0.1, random_state = 1)


# In[10]:


data.shape


# In[11]:


# plot histogram of each parameter... Visualization
data.hist(figsize = (30,30))
plt.show()


# In[12]:


# to determine the number of fraudulent cases
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

fraud_fraction = len(fraud) / float(len(valid))  #ratio of fraud against valid
print(fraud_fraction)


print("Fraud cases : {}".format(len(fraud)))
print("Valid cases : {}".format(len(valid)))


# # Correlation Matrix

# In[13]:


# Correlation matrix

corrmat = data.corr()
fig = plt.figure(figsize = (10,9))

sns.heatmap(corrmat,vmax = 0.8, square = True)
plt.show()


# In[14]:


# to get all the columns of the dataframe
columns = data.columns.tolist()

# filter the columns that we do not want 
columns = [c for c in columns if c not in ["Class"]]

# the variable we'll be predicting on 
target = "Class"

X = data[columns]
Y = data[target]

# print the shape of X and Y
print(X.shape)
print(Y.shape)


# # Algorithms

# In[14]:


from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# In[15]:


# defining the random state
state = 1

# define the fraud detection method
# create a classifier in a dictionary format
classifiers = {"Isolation Forest": IsolationForest(max_samples = len(X),random_state = state, contamination = fraud_fraction),
              "Local Outlier Factor": LocalOutlierFactor(n_neighbors = 20, contamination = fraud_fraction)
    
}


# In[31]:


# fit the model

n_outliers = len(fraud)

for i,(clf_name,clf) in enumerate (classifiers.items()):
    # fit the data and the tag outlier
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
        # Reshape the prediction 0 for valid and 1 for fraudulent
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        
        n_errors = (y_pred != Y).sum()
        
        # Run classification metrics
        print("{} : {}".format(clf_name , n_errors))
        print("Accuracy : " ,accuracy_score(Y , y_pred) * 100)
        print(classification_report(Y , y_pred))


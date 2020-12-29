#!/usr/bin/env python
# coding: utf-8

# In[36]:


import sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split


# In[37]:


train_file="C:/Users/HP/Downloads/a01c26dcd27711ea (1)/Dataset/train.csv"
test_file="C:/Users/HP/Downloads/a01c26dcd27711ea (1)/Dataset/test.csv"


# In[38]:


train_data=pd.read_csv(train_file)
test_data=pd.read_csv(test_file)


# In[39]:


train_data.head()


# In[40]:


features=['condition', 'color_type', 'length(m)', 'height(cm)', 'X1', 'X2']
X=train_data[features]
X_=test_data[features]
y1=train_data[['breed_category']]
y2=train_data[['pet_category']]


# In[41]:


train_X, val_X, train_y, val_y = train_test_split(X, y1, random_state = 0)
train_X, val_X, train_y2, val_y2 = train_test_split(X, y2, random_state = 0)

# Get list of categorical variables
s = (train_X.dtypes == 'object')
object_cols = list(s[s].index)


from sklearn.preprocessing import OneHotEncoder


# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_X[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(val_X[object_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = train_X.index
OH_cols_valid.index = val_X.index


# Remove categorical columns (will replace with one-hot encoding)
num_X_train = train_X.drop(object_cols, axis=1)
num_X_valid = val_X.drop(object_cols, axis=1)
num_X_test = X_.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)


# Number of missing values in each column of training data

from sklearn.impute import SimpleImputer

# Fill in the lines below: imputation
my_imputer = SimpleImputer(strategy='mean') # Your code here
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(OH_X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(OH_X_valid))
imputed_X_test = pd.DataFrame(my_imputer.transform(OH_X_test))


# Fill in the lines below: imputation removed column names; put them back
imputed_X_train.columns = OH_X_train.columns
imputed_X_valid.columns = OH_X_valid.columns
imputed_X_test.columns = OH_X_test.columns


imputed_X_train.head()
#print(reduced_X_train.shape)
#missing_val_count_by_column = (train_X.isnull().sum())
#print(missing_val_count_by_column[missing_val_count_by_column > 0]) '''


# In[42]:


model1=LogisticRegression(C=1e5)
model2=LogisticRegression(C=1e5)
model1.fit(imputed_X_train, train_y)
model2.fit(imputed_X_train, train_y2)


# In[43]:


preds=model1.predict(imputed_X_valid)
preds2=model2.predict(imputed_X_valid)
print(preds)
print(preds2)


# In[47]:



pred=model1.predict(imputed_X_test)
pred2=model2.predict(imputed_X_test)
print(pred)
print(pred2)
print(pred.shape)
print(pred2.shape)


# In[48]:


df = pd.DataFrame({'pet_id':list(test_data['pet_id']),
                    'breed_category':(list(pred)),'pet_category':(list(pred2))})
submission_data2=df
submission_data2.to_csv('final_submission2.csv', index=False)


# In[ ]:





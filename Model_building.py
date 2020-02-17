#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#setting graphical formats
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[3]:


train_df = pd.read_csv('C:/Users/preethi/Downloads/data_train_normal.csv')


# In[200]:


test_df = pd.read_csv('C:/Users/preethi/Downloads/data_test_normal.csv')


# In[195]:


test_df.head()


# In[13]:


train_df.shape


# In[8]:


test_df.shape


# In[12]:


train_df = train_df.drop(["READMIT", "VisitLink","YEAR","found"],axis=1)


# In[14]:


test_df = test_df.drop(["READMIT", "VisitLink","YEAR","found"],axis=1)


# In[15]:


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


# In[161]:


X_train = train_df.drop(['LOS'],axis=1)

y_train = train_df.loc[:,['LOS']]


# In[163]:


X_test = test_df.drop(['LOS'],axis=1)

y_test = test_df.loc[:,['LOS']]


# In[167]:


X_test = X_test[['ADC',
'ADMTOT',
'AGE',
'CHC',
'CM_ANEMDEF',
'CM_COAG',
'CM_LIVER',
'CM_LYTES',
'CM_OBESE',
'CM_PARA',
'CM_WGHTLOSS',
'FTMDTF',
'GENHOS',
'NCHRONIC',
'NDX',
'NPR',
'TOTCHG_X',
'VEM',
'CCI',
'SUROPOP',
'Sepsis',
'UTI',
'PNA',
'DVT',
'Infection',
'MAPP9n',
'HMO86',
'LIVRHOS',
'ATYPE_1',
'ATYPE_2',
'DISPUNIFORM_5.0',
'DISPUNIFORM_6.0',
'HOSPST_CA',
'HOSPST_FL',
'HOSPST_MD',
'HOSPST_WA',
'MEDINCSTQ_1.0',
'PAY1_2.0']]


# In[165]:


y_test.shape


# In[168]:


import statsmodels.api as sm


X_train = X_train[['ADC',
'ADMTOT',
'AGE',
'CHC',
'CM_ANEMDEF',
'CM_COAG',
'CM_LIVER',
'CM_LYTES',
'CM_OBESE',
'CM_PARA',
'CM_WGHTLOSS',
'FTMDTF',
'GENHOS',
'NCHRONIC',
'NDX',
'NPR',
'TOTCHG_X',
'VEM',
'CCI',
'SUROPOP',
'Sepsis',
'UTI',
'PNA',
'DVT',
'Infection',
'MAPP9n',
'HMO86',
'LIVRHOS',
'ATYPE_1',
'ATYPE_2',
'DISPUNIFORM_5.0',
'DISPUNIFORM_6.0',
'HOSPST_CA',
'HOSPST_FL',
'HOSPST_MD',
'HOSPST_WA',
'MEDINCSTQ_1.0',
'PAY1_2.0']]


# Note the difference in argument order
model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test) # make the predictions by the model

# Print out the statistics
model.summary()


# In[184]:


#Main testing of bad cases
d = {'y' : y_test['LOS'], 'pred': np.ceil(predictions)}
prediction = pd.DataFrame(data = d)
tmp = prediction['y']-prediction['pred']


# In[190]:


tmp[tmp<0].count()


# In[191]:


2084/3065
1717/3065


# In[174]:


plt.rcParams["figure.figsize"] = [50,20]
plt.figure(figsize=(500,100))
# ax=plt.gca()
# ax.axis([-25,25,0,8000])
tmp.hist(bins = 100)
plt.show()


# In[49]:


d = {'y' : y_train['LOS'], 'pred': round(predictions,0)}


# In[51]:


prediction = pd.DataFrame(data = d)


# In[54]:


tmp = prediction['y']-prediction['pred']


# In[58]:


tmp.hist(bins = 100)
plt.show()


# In[60]:


model = LinearRegression()
model.fit(X_train, y_train)

predictions1 = model.predict(X_train)

print ('GFT + Wiki / GT R-squared: %.4f' % model.score(X_train, y_train))


# In[72]:


tmp = round(y_train - predictions1,0)


# In[87]:


plt.rcParams["figure.figsize"] = [16,9]
plt.figure(figsize=(500,100))
# ax=plt.gca()
# ax.axis([-25,25,0,8000])
tmp.hist(bins = 100)
plt.show()


# In[82]:


#Cross Validation
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LinearRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


# In[85]:


# Perform 10-fold cross validation
scores = cross_val_score(modelCV, X_train, y_train, cv=10)
print ("Cross validated scores:", scores)


# In[92]:


# from sklearn.base import BaseEstimator, RegressorMixin
# import statsmodels.formula.api as smf
# import statsmodels.api as sm
 
# class statsmodel(BaseEstimator, RegressorMixin):
#     def __init__(self, sm_class, formula):
#         self.sm_class = sm_class
#         self.formula = formula
#         self.model = None
#         self.result = None
 
#     def fit(self,data,dummy):
#         self.model = self.sm_class(self.formula,data)
#         self.result = self.model.fit()
 
#     def predict(self,X):
#         return self.result.predict(X)


# In[227]:


# create training and testing vars
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.2)
print (X_train1.shape, y_train1.shape)
print (X_test1.shape, y_test1.shape)


# In[228]:


model = sm.OLS(y_train1, X_train1).fit()
predictions1 = model.predict(X_test1)


# In[229]:


model.summary()
#80.7, 80.8,80.8,81.5,80.9


# In[230]:


d = {'y' : y_test1['LOS'], 'pred': round(predictions1,0)}
prediction = pd.DataFrame(data = d)
tmp = prediction['y']-prediction['pred']


# In[231]:


print("RMSE",np.sqrt(np.mean(tmp**2)))
print("MSE",np.mean(tmp**2))
print("Absolute Mean",np.mean(np.abs(tmp)))


# In[147]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test1['LOS'], predictions1))


# In[148]:


rms
#4.27, 4.006,4.43,4.17,4.23


# In[116]:


predictions1.shape


# In[1]:


#print ("Score:", model.score(X_test1, y_test1))


# In[96]:



# # create a model
# clf = statsmodel(smf.ols, "LOS ~ ADC+ADMTOT+AGE+CHC+CM_ANEMDEF+CM_COAG+CM_LIVER+CM_LYTES+CM_OBESE+CM_PARA+CM_WGHTLOSS+FTMDTF+GENHOS+NCHRONIC+NDX+NPR+TOTCHG_X+VEM+CCI+SUROPOP+Sepsis+UTI+PNA+DVT+Infection+MAPP9n+HMO86+LIVRHOS+ATYPE_1+ATYPE_2+DISPUNIFORM_5.0+DISPUNIFORM_6.0+HOSPST_CA+HOSPST_FL+HOSPST_MD+HOSPST_WA+MEDINCSTQ_1.0+PAY1_2.0")

# # Print cross val score on this model
# print (cross_val_score(clf, X_train, y_train))

# # # Same thing on sklearn's linear regression model
# # lm = linear_model.LinearRegression()

# # print (cross_val_score(lm , ccard.data.iloc[:,1:3].values, ccard.data.iloc[:,0].values))


# In[27]:


y_train.dtypes


# In[22]:


X_train.dtypes.tail(60)


# In[21]:


X_train.shape


# In[192]:


cost_df = pd.read_csv('C:/Users/preethi/Downloads/cost_df.csv')


# In[193]:


cost_df.head()


# In[206]:


output = test_df[['VisitLink',
'ADC',
'ADMTOT',
'AGE',
'CHC',
'CM_ANEMDEF',
'CM_COAG',
'CM_LIVER',
'CM_LYTES',
'CM_OBESE',
'CM_PARA',
'CM_WGHTLOSS',
'FTMDTF',
'GENHOS',
'NCHRONIC',
'NDX',
'NPR',
'TOTCHG_X',
'VEM',
'CCI',
'SUROPOP',
'Sepsis',
'UTI',
'PNA',
'DVT',
'Infection',
'MAPP9n',
'HMO86',
'LIVRHOS',
'ATYPE_1',
'ATYPE_2',
'DISPUNIFORM_5.0',
'DISPUNIFORM_6.0',
'HOSPST_CA',
'HOSPST_FL',
'HOSPST_MD',
'HOSPST_WA',
'MEDINCSTQ_1.0',
'PAY1_2.0',
'LOS']]


# In[211]:


output['LOS_pred'] = prediction['pred']


# In[213]:


output[['LOS','LOS_pred']]


# In[216]:


output = output.merge(cost_df,on = 'VisitLink',how = 'left')


# In[217]:


output.columns


# In[218]:


#output.columns = output.columns.str.replace('LOS_y', 'Total_READ_LOS')
output.rename(columns={'LOS_y':'Total_READ_LOS',
                        'LOS_pred':'Ideal_LOS',
                        'LOS_x':'Actual_LOS',
                        'TOTCHG_X_y':'Total_READ_CHG'}, 
                 inplace=True)
 


# In[219]:


output.columns


# In[220]:


output['CHG_per_VisitLink'] = output['Total_READ_CHG']/output['Total_READ_LOS']


# In[224]:


output[['CHG_per_VisitLink','Total_READ_CHG','Total_READ_LOS','Ideal_LOS','Cost_Saved']]
#output['Cost_Saved'] = output['CHG_per_VisitLink'] * (output['Total_READ_LOS']-output['Ideal_LOS'])


# In[225]:


output['Cost_Saved'].sum()


# In[209]:


output['LOS']


# In[226]:


output.to_csv('C:/Users/preethi/Downloads/output_final.csv',index=False)







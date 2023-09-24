#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# In[2]:


#reading a csv file
data =pd.read_csv('reallifedata.csv')


# In[3]:


data


# In[4]:


#describe() show the data details only numericals colum
data.describe()


# In[5]:


#include=all describe the all categorical and numerical values
data.describe(include='all')


# In[6]:


#drop is a function just be carefull 
#axis is for row and colums
dataset =data.drop(['Model'],axis=1)
dataset


# In[7]:


dataset.describe(include='all')


# In[8]:


#EDA
#deal with the missing values and show the true and false 
dataset.isnull()


# In[9]:


#sum()  it show the number of truee 
dataset.isnull().sum()


# In[10]:


#drpna is a function in which the true values get erase
data2=dataset.dropna(axis=0)
data2


# In[11]:


data2.describe(include='all')


# In[12]:


sns.distplot(data2['Price'])
plt.show()


# In[13]:


q=data2['Price'].quantile(0.99)
q


# In[14]:


price=data2[data2['Price']<q]
price


# In[15]:


sns.distplot(price['Price'])
plt.show()


# In[16]:


sns.distplot(data2['Mileage'])
plt.show()


# In[17]:


q=price['Mileage'].quantile(0.99)
q


# In[18]:


Mileage=price[price['Mileage']<q]


# In[19]:


sns.distplot(Mileage['Mileage'])
plt.show()


# In[20]:


sns.distplot(Mileage['EngineV'])
plt.show()


# In[21]:


q=Mileage['EngineV'].quantile(0.99)
q


# In[22]:


EngineV=Mileage[Mileage['EngineV']<q]


# In[23]:


sns.distplot(EngineV['EngineV'])
plt.show()


# In[24]:


sns.distplot(EngineV['Year'])
plt.show()


# In[25]:


q=EngineV['Year'].quantile(0.01)
q


# In[26]:


fdata=EngineV[EngineV['Year']>q]
fdata


# In[27]:


sns.distplot(fdata['Year'])
plt.show()


# In[28]:


fdata.describe(include='all')


# In[29]:


#linearity

fdata


# In[30]:


#reseting the index number
data_cleaned=fdata.reset_index(drop=True)


# In[31]:


data_cleaned


# In[32]:


#scattering tpo check the linearity in between mileage and price 
plt.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.show()


# In[33]:


#linearity check to find the 
fig,(ax1,ax2,ax3)=plt.subplots(1,3,sharey=True,figsize=(12,4))
ax1.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax1.set_title('MILEAGE VS PRICE')

ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('ENGINEV VS PRICE')

ax3.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax3.set_title('YEAR VS PRICE')
plt.show()


# In[34]:


#log transformation
#becoz the values could be bigger that the reason to convert intoo the loh values so that we can train the module and at the end we gonna convert into the real values  by the np.exp() method
log_price =np.log(data_cleaned['Price'])
log_price


# In[35]:


sns.distplot(log_price)
plt.show()


# In[36]:


#adding the colum into the data cleaned dataframe
data_cleaned['log_price']=log_price
data_cleaned


# In[37]:


#dropping the pricer colum becoz we can now train our model through the log price
data_cleaned=data_cleaned.drop(['Price'],axis=1)
data_cleaned


# In[38]:


fig,(ax1,ax2,ax3)=plt.subplots(1,3,sharey=True,figsize=(12,4))
ax1.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax1.set_title('MILEAGE VS log_PRICE')

ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('ENGINEV VS log_PRICE')

ax3.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax3.set_title('YEAR VS log_PRICE')
plt.show()


# In[39]:


#SECOND ASSUMPTION 
#MULTICOLINEARITY
test=data_cleaned[['Mileage','EngineV','Year']]
test


# In[40]:


#to test the multicolinearity 
#solution is Vif
#for that we need to impoort statsmodel
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[41]:


test.values


# In[42]:


#A variance inflation factor is a tool to help identify the degree of multicollinearity
vif =[variance_inflation_factor(test.values,i) for i in range(3)]
vif


# In[43]:


# corr() helps companies determine which variables they want to investigate further, and it allows for rapid hypothesis testing. 
test.corr()


# In[44]:


#after analysis the correlation we understand that year is not correlated
data_no_multi=data_cleaned.drop(['Year'],axis=1)


# In[45]:


data_no_multi


# In[46]:


data_no_multi['Brand'].unique()


# In[47]:


data_no_multi['Body'].unique()


# In[48]:


data_no_multi.describe(include='all')


# In[49]:


data_with_dummy=pd.get_dummies(data_no_multi,drop_first=True)


# In[50]:


data_with_dummy


# In[51]:


target=data_with_dummy['log_price']


# In[52]:


target


# In[53]:


inputs=data_with_dummy.drop(['log_price'],axis=1)


# In[54]:


inputs


# In[55]:


from sklearn.preprocessing import StandardScaler


# In[56]:


scaler=StandardScaler()


# In[57]:


scaler.fit(inputs)


# In[58]:


inputs_scaled = scaler.transform(inputs)
inputs_scaled


# In[59]:


sns.distplot(data_with_dummy['Mileage'])


# In[60]:


test_df=pd.DataFrame(inputs_scaled)
test_df


# In[61]:


sns.distplot(test_df[0])


# In[62]:


inputs_scaled.shape

target.shape
# In[63]:


target.shape


# In[64]:


# 80 20 rule
from sklearn.model_selection import train_test_split


# In[68]:


X_train,x_test,Y_train,y_test = train_test_split(inputs_scaled,target,test_size=0.2,random_state=0)


# In[69]:


X_train.shape


# In[71]:


x_test.shape


# In[74]:


from sklearn.linear_model import LinearRegression


# In[75]:


model= LinearRegression()


# In[76]:


model.fit(X_train,Y_train)


# In[77]:


model.score(X_train,Y_train)


# In[78]:


model.score(x_test,y_test)


# In[80]:


y_pred =model.predict(x_test)


# In[81]:


y_pred


# In[83]:


d=pd.DataFrame(y_pred)


# In[84]:


d


# In[86]:


plt.scatter(y_test,y_pred)
plt.xlabel('actual value')
plt.ylabel('predicted value')
plt.show()


# In[87]:


y_pred=np.exp(y_pred)


# In[89]:


d=pd.DataFrame(y_pred)


# In[91]:


d


# In[92]:


x_test


# In[ ]:





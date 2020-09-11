#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing,
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import datetime


# In[2]:


items_cat = pd.read_csv("item_categories.csv")
items = pd.read_csv("items.csv")
train = pd.read_csv("sales_train.csv")
test = pd.read_csv("test.csv")
shop = pd.read_csv("shops.csv")
#submission = pd.read_csv("sample_submission.csv")


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
flierprops = dict(marker='o', markerfacecolor='green', markersize=6,
                  linestyle='none', markeredgecolor='yellow')
sns.boxplot(x=train.item_cnt_day, flierprops=flierprops)

plt.figure(figsize=(10,4))
plt.xlim(train.item_price.min(), train.item_price.max()*1.1)
sns.boxplot(x=train.item_price, flierprops=flierprops)


# In[6]:


train = train[(train.item_price < 300000 )& (train.item_cnt_day < 1000)]


# In[7]:


median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()
train.loc[train.item_price<0, 'item_price'] = median


# In[8]:


train.loc[train.item_cnt_day < 1, "item_cnt_day"] = 0


# In[9]:


train = train.rename(columns={'date':'DATE'})
train['DATE'] = pd.to_datetime(train['DATE'], format='%d.%m.%Y')
train['month'] = train['DATE'].dt.month
train['year'] = train['DATE'].dt.year


# In[10]:


train = train.drop(['DATE'], axis=1)


# In[11]:


items.head()


# In[12]:


shop.head()


# In[13]:


items_cat.head()


# In[14]:


items['item_name_length'] = items['item_name'].map(lambda x : len(x)) 
items['item_name_word_count'] = items['item_name'].map(lambda x : len(x.split(' ')))


# In[15]:


items_cat['item_categories_name_length'] = items_cat['item_category_name'].map(lambda x : len(x)) 
items_cat['item_categories_name_word_count'] = items_cat['item_category_name'].map(lambda x : len(x.split(' ')))


# In[16]:


shop['shop_name_length'] = shop['shop_name'].map(lambda x : len(x)) 
shop['shop_name_word_count'] = shop['shop_name'].map(lambda x : len(x.split(' ')))


# In[17]:


items.head()


# In[18]:


shop.head()


# In[19]:


items_cat.head()


# In[20]:


train.info()


# In[21]:


test.info()


# In[22]:


train = train.drop([ 'item_price'], axis=1)
train = train.groupby([c for c in train.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train = train.rename(columns={'item_cnt_day':'item_cnt_month'})


# In[23]:


train.head()


# In[24]:


#Monthly mean
SIMA = train[['shop_id', 'item_id', 'item_cnt_month']].groupby(['shop_id', 'item_id'], as_index=False)[['item_cnt_month']].mean()
SIMA = SIMA.rename(columns={'item_cnt_month':'item_cnt_month_mean'})

#Add Mean Features
train = pd.merge(train, SIMA,how='left', on=['shop_id', 'item_id'])
train


# In[25]:


train = pd.merge(train, items, how='left', on='item_id')
train.head()


# In[26]:


train.shape


# In[27]:


train = pd.merge(train, items_cat, how='left', on=['item_category_id'])
train.head()


# In[28]:


train.shape


# In[29]:


train = pd.merge(train, shop, how='left', on=['shop_id'])
train.head()


# In[30]:


train.shape


# In[31]:


#NOW test data


# In[32]:


test['month'] = 11
test['year'] = 2015
test['date_block_num'] = 34
test.head()


# In[33]:


test.shape


# In[34]:


SIMA.head()


# In[35]:


SIMA.shape


# In[36]:


test = pd.merge(test, SIMA, how='left', on=['shop_id', 'item_id'])
print(len(test))


# In[37]:


test.head()


# In[38]:


test = pd.merge(test, items, how='left', on='item_id')
test.head()


# In[39]:


test.shape


# In[40]:


test = pd.merge(test, items_cat, how='left', on='item_category_id')
test.shape


# In[41]:


test = pd.merge(test, shop, how='left', on='shop_id')
test.shape


# In[42]:


test['item_cnt_month'] = 0.
test.shape


# In[43]:


test.head()


# In[44]:


train.shape


# In[45]:


train.head()


# In[46]:


import sklearn
import nltk


# In[47]:


for c in ['shop_name', 'item_category_name', 'item_name']:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(list(train[c].unique()) + list(test[c].unique()))
    train[c] = le.transform(train[c].astype(str))
    test[c] = le.transform(test[c].astype(str))
    print(c)


# In[48]:


train


# In[49]:


test


# In[50]:


feature_list = [c for c in train.columns if c not in 'item_cnt_month']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[51]:


from sklearn.ensemble import RandomForestRegressor


# In[52]:


rf = RandomForestRegressor(n_estimators=25, random_state=42, max_depth=15, n_jobs=-1)


# In[ ]:





# In[53]:


def algo_package(chunk,train_df):
    x1 = train_df[train_df['date_block_num'] < chunk]
    y1 = np.log1p(x1['item_cnt_month'].clip(0., 20.))
    x1 = x1[feature_list]
    x2 = train_df[train_df['date_block_num'] == chunk]
    y2 = np.log1p(x2['item_cnt_month'].clip(0., 20.))
    x2 = x2[feature_list]
    rf.fit(x1, y1)
    y2hat =rf.predict(x2)
    val_rmse =    np.sqrt(sklearn.metrics.mean_squared_error(y2.clip(0., 20.), y2hat.clip(0., 20.)))
    val_rsqaure =  sklearn.metrics.r2_score(y2.clip(0., 20.), y2hat.clip(0., 20.))
    out = [val_rmse,val_rsqaure]
    print("RMSE on :",chunk,"th date_block_num is :",val_rmse)
    print("R2 on : ",chunk,"th dateblock_num is",val_rsqaure)                          
    return out





# In[54]:


i=1
mat=[]
while(5*i<33):
        chunk=5*i
        error=algo_package(chunk,train)
        mat.append([chunk,error])
        i=i+1

        
        


# In[ ]:





# In[ ]:





# In[59]:


#Full train
feature_list = [c for c in train.columns if c not in 'item_cnt_month']
rf = RandomForestRegressor(n_estimators=25, random_state=42, max_depth=15, n_jobs=-1)
rf.fit(train[feature_list], train['item_cnt_month'].clip(0., 20.))
print("Accuracy on training data without considering variable importances:{}".format(round(rf.score(train[feature_list], train['item_cnt_month'].clip(0., 20.))*100, 2)))


# In[57]:


importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[58]:


# Set the style
plt.style.use('fivethirtyeight')

#set size
fig = plt.figure(figsize=(25, 5))
ax = fig.add_subplot(111)

# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
ax.bar(x_values, importances, orientation = 'vertical')

# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical', fontsize=20)

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[61]:


rf_most_important = RandomForestRegressor(n_estimators=25, random_state=42, max_depth=15, n_jobs=-1)

# Extract the ten most important features
important_features = ['item_cnt_month_mean', 'date_block_num', 'item_id', 'item_name', 'item_name_length', 'year','item_name_word_count', 'item_category_id' ,'item_category_name','item_categories_name_length','shop_id']

#Full train
rf_most_important.fit(train[important_features], train['item_cnt_month'].clip(0., 20.))
print("Accuracy on training data considering variable importances:{}".format(round(rf_most_important.score(train[important_features], train['item_cnt_month'].clip(0., 20.))*100, 2)))


# In[62]:


test.head(10)


# In[64]:


test = test.fillna(0.)
test['item_cnt_month'] = 0.
test.head()


# In[65]:


test['item_cnt_month'] = rf_most_important.predict(test[important_features]).clip(0., 20.)


# In[66]:


test[['ID', 'item_cnt_month']].to_csv('submission_rf.csv', index=False)


# In[ ]:





# In[ ]:





# In[67]:


rf.fit(x1,y1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[70]:


import pickle


# In[71]:


filename = 'random_forest_picklefile'
outfile = open(filename,'wb')


# In[72]:


pickle.dump(rf_most_important,outfile)
outfile.close()


# In[ ]:





# In[73]:


infile = open(filename,'rb')
RF_new = pickle.load(infile)
infile.close()


# In[74]:


RF_new


# In[75]:


rf_most_important


# In[76]:


from joblib import dump,load


# In[77]:


dump(rf_most_important,'future_sales_predict_kaggle.joblib')


# In[ ]:





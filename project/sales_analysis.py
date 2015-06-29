# -*- coding: utf-8 -*-
""" 
This analysis was created for a proprietary data set that belongs to a major 
pharmaceutical company. The goal was to segment their medium-to-small customer
base and to determine which customers have responded best to price reduction
campaigns.

The main dataframe consists of customer code, item number, weekly sales 
and weekly returns for 2014. A secondary dataframe consists of the dates 
when price reductions campaigns were held for paticular items for each customer. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import seaborn as sns
import pickle
from scipy import stats

#%% Read in data (file names redacted)

db =  pd.read_csv('...', sep = ',') #main data frame

#price reduction campaigns converted to week of the year for each customer/item
campaign_dates = pd.read_excel('...',sheetname=0)
campaign_dates['start_week']=campaign_dates['start_dt'].dt.weekofyear
campaign_dates['end_week']=campaign_dates['end_dt'].dt.weekofyear
campaign_dates['end_week'].replace(to_replace=1, value=52, inplace=True)
campaign_dates.columns=['CUST_NUM','ITEM_NUM','START_DATE','END_DATE','START_WEEK','END_WEEK']

#%% cluster customers by weekly sales

#get weekly sales matrix and remove customers with missing data
by_cust_week = db.groupby(['CUST_NUM','W'],as_index=False).sum()
sales_series_cust=by_cust_week.pivot(index='W', columns='CUST_NUM', values='SLSAMT')
sales_series_cust.fillna(0,inplace=True)
sales_series_cust2 = sales_series_cust.loc[:, (sales_series_cust != 0).any(axis=0)]

#keep track of which customers were removed
all_customers=db['CUST_NUM'].unique()
cust_to_remove=(sales_series_cust.sum() == 0).nonzero()[0]
customers_used = np.delete(all_customers,cust_to_remove) 

#cluster using covariance matrix as similarity matrix
sales_cov=sales_series_cust2.cov()
_, cust_labels = cluster.affinity_propagation(sales_cov)

#keep cluster information
cust_to_group = dict(zip(customers_used, cust_labels))

cluster_dictionary = {} #maps clusters to all customers (index in customers_used) in them
for index,label in enumerate(cust_labels):
    if label not in cluster_dictionary:
        cluster_dictionary[label]=[index]
    else:
        cluster_dictionary[label].append(index)

main_clusters=[] #clusters with more than one customer        
for key in cluster_dictionary.keys():
    if len(cluster_dictionary[key]) > 1:
        main_clusters.append((key, len(cluster_dictionary[key]) ,customers_used[cluster_dictionary[key][0]]))

main_cluster_number = [x[0] for x in main_clusters]

main_mean_group_sales=[] #mean sales trend per customer
main_ste_group_sales=[] #standard error on sales trend per customer       
for group in main_cluster_number:
    locations = cluster_dictionary.get(group)
    selected_data = sales_series_cust2.iloc[:,(locations)]
    group_mean = np.mean(selected_data,axis=1)
    group_ste = stats.sem(selected_data,axis=1)
    main_mean_group_sales.append(group_mean)
    main_ste_group_sales.append(group_ste)     

main_mean_group_sales=np.asarray(main_mean_group_sales)        
main_ste_group_sales=np.asarray(main_ste_group_sales)



#%%cluster items by weekly sales

#get weekly sales matrix and remove items with missing data
by_item_week = db.groupby(['ITEM_NUM','W'],as_index=False).sum()
sales_series_item=by_item_week.pivot(index='W', columns='ITEM_NUM', values='SLSAMT')
sales_series_item.fillna(0,inplace=True)
sales_series_item2 = sales_series_item.loc[:, (sales_series_item != 0).any(axis=0)]

#keep track of which items were removed
all_items=db['ITEM_NUM'].unique()
items_to_remove=(sales_series_item.sum() == 0).nonzero()[0]
items_used = np.delete(all_items,items_to_remove) 

#cluster using covariance matrix as similarity matrix
item_sales_cov=sales_series_item2.cov()

_, item_labels = cluster.affinity_propagation(item_sales_cov)

#keep cluster information
item_to_group = dict(zip(items_used, item_labels))

item_dictionary = {} #maps clusters to all items (index in item_used) in them
for index,label in enumerate(item_labels):
    if label not in item_dictionary:
        item_dictionary[label]=[index]
    else:
        item_dictionary[label].append(index)
 
item_examples=[]
for label in set(item_labels):
    item_examples.append(items_used[item_dictionary[label][0]])
    
#%% calculate customer response to sales 

merged_dates = pd.merge(db, campaign_dates, on = ['CUST_NUM','ITEM_NUM'], how='left')

# go through marking items that were on sale 
merged_dates['SALE']=0
merged_dates.loc[(merged_dates['START_WEEK'] <= merged_dates['W']) & (merged_dates['W'] <= merged_dates['END_WEEK']), 'SALE'] = 1
merged_dates.drop(['START_DATE','END_DATE','START_WEEK','END_WEEK'],axis=1,inplace=True)

merged_dates.sort('SALE',inplace=True)

#drop the additional SALE=0 row that appears if item was ever on sale
merged_dates.drop_duplicates(subset=['CUST_NUM', 'ITEM_NUM','W'], take_last=True,inplace=True) 

#make data frame that keeps average weekly sales per item per customer on vs off sale
grouped=merged_dates.groupby(['CUST_NUM','ITEM_NUM','SALE'],as_index=False).mean()
grouped=grouped.groupby(['CUST_NUM','ITEM_NUM']).filter(lambda x: len(x) > 1)
compare_sales=pd.DataFrame()
compare_sales['CUST_NUM']=grouped['CUST_NUM'].loc[grouped['SALE']==1]
compare_sales['WITH']=grouped['SLSAMT'].loc[grouped['SALE']==1]
compare_sales['WITHOUT']=np.asarray(grouped['SLSAMT'].loc[grouped['SALE']==0])
compare_sales = compare_sales[~(compare_sales == 0).any(axis=1)]

#proportion of weekly sales on sale to off sale (per item)
compare_sales['PROP']=compare_sales['WITH']/compare_sales['WITHOUT']
compare_sales=compare_sales.groupby('CUST_NUM').mean()


#%% summary variables for app

clust_max_sales=[] #customer cluster with the highest average sales for each week  
for week in range(1,53):
    loc=week-1    
    max_loc=np.argmax(main_mean_group_sales[:,loc])
    clust_max_sales.append(main_cluster_number[max_loc])
    

cust_max_score=[] #customers with highest response score for example clusters 
max_scores=[] #scores for these customers
for clust in clust_max_sales:
    customer_loc=cluster_dictionary.get(clust)
    customers=customers_used[customer_loc]
    props=compare_sales['PROP'].loc[customers]
    props.sort(ascending=False,inplace=True)
    if len(props[props>1])>10:
        cust_max_score.append(props[props>1].index[:10])
        max_scores.append(props[props>1][:10].values)
    else:
        cust_max_score.append(props[props>1].index)
        max_scores.append(props[props>1].values)

max_scores=np.asarray(max_scores)    
max_scores=[['{:.2f}'.format(x) for x in array] for array in max_scores]

#table of total yearly sales (customer by item)
by_cust_item=db.groupby(['CUST_NUM','ITEM_NUM'],as_index=False).sum()  
by_cust_item.drop(['W','SLSQTY','RETURNQTY','RETURNAMT','TOTQTY','TOTAMT'],axis=1,inplace=True)
item_sales_table=by_cust_item.pivot(index='CUST_NUM', columns='ITEM_NUM', values='SLSAMT')    

np.asarray(item_sales_table)  
 
#%% save variables of interest to file
  
with open('customer_list.pickle', 'wb') as handle:
  pickle.dump(customers_used, handle)
  
with open('item_list.pickle', 'wb') as handle:
  pickle.dump(all_items, handle)

with open('main_mean_group_sales.pickle', 'wb') as handle:
  pickle.dump(main_mean_group_sales, handle)        

with open('main_cluster_number.pickle', 'wb') as handle:
  pickle.dump(main_cluster_number, handle)

with open('cust_max_score.pickle', 'wb') as handle:
  pickle.dump(cust_max_score, handle)

with open('max_scores.pickle', 'wb') as handle:
  pickle.dump(max_scores, handle)
      
with open('clust_max_sales.pickle', 'wb') as handle:
  pickle.dump(clust_max_sales, handle)  
  
with open('cust_by_item_table.pickle', 'wb') as handle:
  pickle.dump(item_sales_table, handle)
  
#%% various plots



#sales trends for example clusters
plt.figure()
sns.set(font_scale=2)
for row in [0,2,3,5,11]:     
    plt.plot(range(1,53),main_mean_group_sales[row,:])
plt.xlabel('Week')
plt.ylabel('Sales (USD)')
plt.title('Weekly sales for example customer groups')

#histogram of customer response score
prop=np.asarray(compare_sales['PROP']) 
plt.figure()
sns.set(font_scale=2) 
plt.hist(prop,bins=50,range=(0,10))
plt.xlabel('Response score')
plt.ylabel('Frequency')

#sales trends for a few members of each cluster
for clust in main_cluster_number:
    customer_loc=cluster_dictionary.get(clust)[:10]
    plt.figure()
    sns.set(font_scale=2)
    for i in customer_loc:     
        plt.plot(range(1,53),sales_series_cust2.iloc[:,i])
        plt.xlim(0,53)
        plt.xlabel('Week')
        plt.ylabel('Sales (USD)')
        plt.title('Weekly sales for cluster number '+ str(clust))
        
    

    
    
   
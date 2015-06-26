
from flask import render_template, request
from app import app
from a_Model import ModelIt
import pickle
import matplotlib.pyplot as plt
import seaborn
import StringIO
import numpy as np

@app.route('/')
@app.route('/index')
def index():
	return render_template("input.html")
  
@app.route('/about')
def ch_about():
  return render_template("about.html")
  
@app.route('/input')
def ch_input():
  return render_template("input.html")

@app.route('/output')
def ch_output():
  week = int(request.args.get('WEEK'))
  item = int(request.args.get('ITEM'))
  cust_max_score = pickle.load(open("/home/ubuntu/app/cust_max_score.pickle", "rb"))
  max_scores = pickle.load(open("/home/ubuntu/app/max_scores.pickle", "rb"))
  clusters = pickle.load(open("/home/ubuntu/app/clust_max_sales.pickle", "rb"))    
  cust_by_item_table = np.asarray(pickle.load(open("/home/ubuntu/app/cust_by_item_table.pickle", "rb")))
  customer_list = pickle.load(open("/home/ubuntu/app/customer_list.pickle", "rb"))
  item_list = pickle.load(open("/home/ubuntu/app/item_list.pickle", "rb"))    
  item_loc=np.where(item_list==item)[0][0] #index for selected item
  clust = clusters[week-1] #cluster with highest sales for selected week
  cluster_list = pickle.load(open("/home/ubuntu/app/main_cluster_number.pickle", "rb"))    
  cluster_loc=np.where(cluster_list==clust)[0][0] #index for relevant cluster
  week_cust=cust_max_score[week-1] #customers with highest response score in cluster
  cust_loc=[np.where(customer_list==x)[0][0] for x in week_cust] #index for these customers
  week_scores=max_scores[week-1] #scores for these customers
  max_len=5
  #to remove extra rows in table if few customers
  if len(week_cust)<max_len:
      max_len=len(week_cust)
  #get customer with maximum sales for selected item
  cust_item_sales=cust_by_item_table[cust_loc,item_loc]
  max_sales_loc=np.argmax(cust_item_sales)
  #add description to week numbers
  week_num = np.array([1,4,5,6,17,18,34,38,44,47,51])
  week_loc = np.where(week_num==week)[0][0]
  week_str = ['Jan. 1', 'Jan. 22', 'Jan. 29', 'Feb. 5', 'Apr 23', 'Apr. 30', 'Aug. 20', 
             'Sep. 17', 'Oct. 29', 'Nov. 19', 'Dec. 17']  
  plotstr = ModelIt(cluster_loc) #weekly sales plot for full year
  return render_template("output.html", customers = week_cust, scores = week_scores, max_val=max_len, week = week, cluster=clust, max_customer=week_cust[max_sales_loc], item=item, results=plotstr, date_str=week_str[week_loc])

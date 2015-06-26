# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:28:04 2015

@author: noga
"""
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ModelIt(code):
  #plots weekly sales for full year for given customer cluster  
  mean_group_sales = pickle.load(open("/home/ubuntu/app/main_mean_group_sales.pickle", "rb"))      
  plt.figure()
  plt.plot(range(1,53),mean_group_sales[code,:])
  plt.xlim(0,53)
  plt.ylim(0,)  
  plt.xlabel('Week')
  plt.ylabel('Sales (USD)')
  #to export figure  
  from io import BytesIO
  figfile = BytesIO()
  plt.savefig(figfile, format='png')
  figfile.seek(0)
  import base64
  figdata_png = base64.b64encode(figfile.getvalue())
  return figdata_png
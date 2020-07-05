# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 19:10:49 2020

@author: Rohith
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import scale

data=pd.read_csv("E:\\Data Science\\Data_Science_Projects\\Project_6_Predicting_Car_selling_Price\\Dataset\\data.csv")
data=data.drop(["Car_Name"],axis=1) 
data["Current_year"]=2020
data["no_year"]=data["Current_year"]-data["Year"]
#Now drop Current_Year and Year column in data set
data_new=pd.get_dummies(data,columns={"Fuel_Type","Seller_Type","Transmission"},drop_first=True)
df=data_new.drop(['Year','Current_year'],axis=1)

X=df.iloc[:,1:9]
y=df.iloc[:,0]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size= 0.2)

from sklearn.ensemble import GradientBoostingRegressor
model=GradientBoostingRegressor(learning_rate=0.15,n_estimators=200,min_samples_split=10,min_samples_leaf= 1,max_features= 'auto',max_depth= 10)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

pickle.dump(model,open('Model.pkl','wb'))







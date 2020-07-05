# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 17:29:14 2020

@author: Rohith.N
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("E:\\Data Science\\Data_Science_Projects\\Project_6_Predicting_Car_selling_Price\\Dataset\\data.csv")

#############################Exploratory Data Analysis############################################################

data.head() # To show first 5 rows of the data set

data.tail()# To show last 5 rows of the data set

data.columns# To show the number of columns in the data set

data.isnull().sum()#To check is their any null values in the features of the data set

data.dtypes # To check the data types of the features in data set

data.describe() #Give statistical information about the data set such as mean ,median ,mode,max value,min value

data.info() #Give information about any null values/missing value

#Since we are not considering Car_Name .So we are dropping tat column

data=data.drop(["Car_Name"],axis=1) 

print(data)

# To find the number of categorical features in data set

categorical_features=[feature for feature in data.columns if data[feature].dtypes == 'O']
for features in categorical_features:
    print(features)
    
print(len(categorical_features))

# To find the number of numeric feature in the data set

numerical_feature=[feature for feature in data.columns if data[feature].dtypes != 'O']
for features in numerical_feature:
    print(features)
    
print(len(numerical_feature))

## Year feature we have to convert into numeric

data["Current_year"]=2020

data["no_year"]=data["Current_year"]-data["Year"]

#Now drop Current_Year and Year column in data set

data_new=pd.get_dummies(data,columns={"Fuel_Type","Seller_Type","Transmission"},drop_first=True)

df=data_new.drop(['Year','Current_year'],axis=1)

df.head()

df.shape

cormat=df.corr()#Helps to understand correlation between the dependent and independent variables

############################Data vizualization###########################################################

#To plot histogram plot

for features in df:  #it shows the histogram plot for all features which are numeric
    data=df.copy()
    data[features].hist(bins=25)
    plt.title(features)
    plt.xlabel(features)
    plt.ylabel('count')
    plt.show()
    
    # or we can use below plot
    
    df.hist(figsize=(12,12))
    
#To draw the plots for categorical input data
    
    
categorical_features=[feature for feature in df.columns if df[feature].dtypes == 'O']

for feature in categorical_features:
    data=data.copy()
    data.groupby(feature)['Selling_Price'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Selling_Price')
    plt.title(feature)
    plt.show()


df.plot(kind='box',subplots=True,layout=(4,4),sharex=False,sharey=False,figsize=(18,18))


import seaborn as sns

sns.pairplot(df) #To draw pairs plot



#To draw heatmap :
cormat=df.corr()
fig= plt.figure(figsize=(12,12))
sns.heatmap(cormat,annot=True,cmap="BuGn_r")
plt.show()

#or we can use below plots
cormat=df.corr()
sns.heatmap(cormat)

#To draw joint plot using seaborn function
sns.jointplot(x='Kms_Driven',y='Selling_Price',data=df,kind='reg')
sns.jointplot(x='Present_Price',y='Selling_Price',data=df,kind='reg')

sns.distplot(df['Selling_Price'])
sns.distplot(df['Present_Price'])
sns.distplot(df['Kms_Driven'])
sns.distplot(df['Fuel_Type'])
sns.distplot(df['Owner'])


#Below are the plots for categorical data 

# To draw the countplot using seaborn library

sns.countplot('Fuel_Type',data=data,palette='hls')
sns.countplot('Seller_Type',data=data,palette='hls')
sns.countplot('Transmission',data=data,palette='hls')


#To draw the boxplot using seaborn library

sns.boxplot(x='Fuel_Type',y='Selling_Price',data=data,palette='hls')
sns.boxplot(x='Seller_Type',y='Selling_Price',data=data,palette='hls')
sns.boxplot(x='Transmission',y='Selling_Price',data=data,palette='hls')


#To draw violin plot

sns.violinplot(x="Fuel_Type", y="Selling_Price", data=data,palette='rainbow')
sns.violinplot(x="Seller_Type", y="Selling_Price", data=data,palette='rainbow')
sns.violinplot(x="Transmission", y="Selling_Price", data=data,palette='rainbow')


#########################To check the feature importance ######################################################

print(df)

X=df.iloc[:,1:9]
y=df.iloc[:,0]

from sklearn.ensemble import ExtraTreesRegressor

model=ExtraTreesRegressor()

model.fit(X,y)

print(model.feature_importances_)


#Plotting the graph of important features for better vizualizations

feature_importance= pd.Series(model.feature_importances_,index=X.columns)
feature_importance.nlargest(5).plot(kind='barh')
plt.show()

############################Model_Building#######################################################################

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size= 0.3)

#1)Random_Forest_Regressor

from sklearn.ensemble import RandomForestRegressor

help(RandomForestRegressor)


model=RandomForestRegressor()

####################Steps for hyperparameter tuning########################################################
from sklearn.model_selection import RandomizedSearchCV

help(RandomizedSearchCV)


##Creating Random grid

random_grid={'n_estimators':[100,200,300,400,500,600,700,800,900,1000,1100,1200],
             'max_depth':[5,10,15,20,25,30],
             'max_features':['auto','sqrt'],
             'min_samples_split':[2,5,10,15,100],
             'min_samples_leaf':[1,2,5,10]}

rf_random=RandomizedSearchCV(estimator=model,param_distributions=random_grid,scoring='neg_mean_squared_error',n_jobs=1,
                                         verbose=2,cv=5,random_state=42)

rf_random.fit(X_train,y_train)

rf_random.best_params_


###So the best values for  the hyperparametes {'n_estimators': 700,'min_samples_split': 15,'min_samples_leaf': 1,'max_features': 'auto','max_depth': 20} 
#we get from RandomizedSearchCV


model1=RandomForestRegressor(n_estimators=1000,min_samples_split=2,min_samples_leaf= 1,max_features= 'sqrt',max_depth= 25)

model1.fit(X_train,y_train)

y_pred=model1.predict(X_test)

print(y_pred)

sns.distplot(y_test-y_pred)

plt.scatter(y_test,y_pred);plt.xlabel('y_test');plt.ylabel('y_pred')

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(model1, X, y, cv=cv)

from sklearn.metrics import r2_score

print(r2_score(y_test,y_pred))

#########################################################################################################

# Xgboost regressor

import xgboost as xgb

model2=xgb.XGBRegressor()

help(xgb.XGBRegressor())

###########To find the best parameters for model building with the help of hyperparameter tuning###########


from sklearn.model_selection import RandomizedSearchCV

params={'learning_rate':[0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
        'n_estimators':[100,200,300,400,500,600,700,800,900,1000,1100,1200],
             'max_depth':[5,10,15,20,25,30],
             'max_features':['auto','sqrt'],
             'min_samples_split':[2,5,10,15,100],
             'min_samples_leaf':[1,2,5,10]}

xgb_random=RandomizedSearchCV(estimator=model2,param_distributions=params,scoring='neg_mean_squared_error',n_jobs=1,
                                         verbose=2,cv=5,random_state=42)

xgb_random.fit(X_train,y_train)

xgb_random.best_params_

###So the best values for  the hyperparametes {'n_estimators': 400,'min_samples_split': 5,'min_samples_leaf': 10,'max_features': 'sqrt','max_depth': 15,'learning_rate': 0.3} 
#we get from RandomizedSearchCV



model3=xgb.XGBRegressor(learning_rate=0.1,n_estimators=300,min_samples_split=2,min_samples_leaf= 2,max_features= 'auto',max_depth= 30)

model3.fit(X_train,y_train)

y_pred1=model3.predict(X_test)

print(y_pred1)

sns.distplot(y_test-y_pred1)

plt.scatter(y_test,y_pred1);plt.xlabel('y_test');plt.ylabel('y_pred')

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred1))
print('MSE:', metrics.mean_squared_error(y_test, y_pred1))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,y_pred1)))


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(model3, X, y, cv=cv)

from sklearn.metrics import r2_score

print(r2_score(y_test,y_pred1))


#########################################################################################################################

#GradientBoostingRegressor Method

from sklearn.ensemble import GradientBoostingRegressor

model4=GradientBoostingRegressor()

help(GradientBoostingRegressor())


###########To find the best parameters for model building with the help of hyperparameter tuning###########


from sklearn.model_selection import RandomizedSearchCV

params1={'learning_rate':[0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
        'n_estimators':[100,200,300,400,500,600,700,800,900,1000,1100,1200],
             'max_depth':[5,10,15,20,25,30],
             'max_features':['auto','sqrt'],
             'min_samples_split':[2,5,10,15,100],
             'min_samples_leaf':[1,2,5,10]}

gbr_random=RandomizedSearchCV(estimator=model4,param_distributions=params1,scoring='neg_mean_squared_error',n_jobs=1,
                                         verbose=2,cv=5,random_state=42)

gbr_random.fit(X_train,y_train)

gbr_random.best_params_

###So the best values for  the hyperparametes {'n_estimators': 200,'min_samples_split': 10,'min_samples_leaf': 1,'max_features': 'auto','max_depth': 10,'learning_rate': 0.15} 
#we get from RandomizedSearchCV



model5=GradientBoostingRegressor(learning_rate=0.15,n_estimators=200,min_samples_split=10,min_samples_leaf= 1,max_features= 'auto',max_depth= 10)

model5.fit(X_train,y_train)

y_pred2=model5.predict(X_test)

print(y_pred2)

sns.distplot(y_test-y_pred2)

plt.scatter(y_test,y_pred2);plt.xlabel('y_test');plt.ylabel('y_pred')

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred2))
print('MSE:', metrics.mean_squared_error(y_test, y_pred2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,y_pred2)))


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(model5, X, y, cv=cv)

from sklearn.metrics import r2_score

print(r2_score(y_test,y_pred2))

#######################################################################################################################################################################

#DecisionTree Regressor

from sklearn.tree import DecisionTreeRegressor

model6=DecisionTreeRegressor()

help(DecisionTreeRegressor())


###########To find the best parameters for model building with the help of hyperparameter tuning###########


from sklearn.model_selection import RandomizedSearchCV

params2={'criterion':['mse','friedman_mse', 'mae'],
         'splitter':['best','random'],
             'max_depth':[5,10,15,20,25,30],
             'max_features':['auto','sqrt'],
             'min_samples_split':[2,5,10,15,100],
             'min_samples_leaf':[1,2,5,10]}

dec_random=RandomizedSearchCV(estimator=model6,param_distributions=params2,scoring='neg_mean_squared_error',n_jobs=1,
                                         verbose=2,cv=5,random_state=42)

dec_random.fit(X_train,y_train)

dec_random.best_params_

###So the best values for  the hyperparametes {'splitter =best':'min_samples_split':2,'min_samples_leaf': 2,'max_features': 'auto','max_depth': 25,'criterion': mse} 
#we get from RandomizedSearchCV



model7=DecisionTreeRegressor(criterion='mse',splitter='best',min_samples_split=2,min_samples_leaf= 2,max_features= 'auto',max_depth= 25)

model7.fit(X_train,y_train)

y_pred3=model7.predict(X_test)

print(y_pred3)

sns.distplot(y_test-y_pred3)

plt.scatter(y_test,y_pred3);plt.xlabel('y_test');plt.ylabel('y_pred')

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred3))
print('MSE:', metrics.mean_squared_error(y_test, y_pred3))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,y_pred3)))


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(model7, X, y, cv=cv)

from sklearn.metrics import r2_score

print(r2_score(y_test,y_pred3))


##########################################################################################################################

#Adaboost Regressor

from sklearn.ensemble import AdaBoostRegressor

model8=AdaBoostRegressor()

help(AdaBoostRegressor())


###########To find the best parameters for model building with the help of hyperparameter tuning###########


from sklearn.model_selection import RandomizedSearchCV

params3={'learning_rate':[0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
        'n_estimators':[100,200,300,400,500,600,700,800,900,1000,1100,1200],
        'loss':['linear', 'square', 'exponential']}

ada_random=RandomizedSearchCV(estimator=model8,param_distributions=params3,scoring='neg_mean_squared_error',n_jobs=1,
                                         verbose=2,cv=5,random_state=42)

ada_random.fit(X_train,y_train)

ada_random.best_params_

###So the best values for  the hyperparametes {'loss' ='square',learning_rate=0.3,n_estimators = 1000}
#we get from RandomizedSearchCV



model9=AdaBoostRegressor(learning_rate=0.3,n_estimators=1000,loss='square')

model9.fit(X_train,y_train)

y_pred4=model9.predict(X_test)

print(y_pred4)

sns.distplot(y_test-y_pred4)

plt.scatter(y_test,y_pred4);plt.xlabel('y_test');plt.ylabel('y_pred')

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred3))
print('MSE:', metrics.mean_squared_error(y_test, y_pred3))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,y_pred3)))


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(model9, X, y, cv=cv)

from sklearn.metrics import r2_score

print(r2_score(y_test,y_pred3))

#########################################################################################################################


########From all the above regression models we can find GradientBoostingRegressor is having better accuracy values,RMSE Values compared to all others
##########So the finalised model is GradientBoostingRegressor model#######################################




































































































































































































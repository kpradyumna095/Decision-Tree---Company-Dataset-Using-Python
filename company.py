# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:07:33 2019

@author: Hello
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

company = pd.read_csv("C:\\Users\\Hello\\Desktop\\Data science\\data science\\assignments\\DT & RF\\datasets\\Company_Data.csv")

##Checking for maximum and minimum values to decide what will be the cut off point
company["Sales"].min()
company["Sales"].max()
company["Sales"].value_counts()

##Converting it Sales variable into categorical data i.e we have bucketed the data into two levels.
## Less than 7.5 and greater than 7.5

##Knowing the middle value by looking into median so that i find the middle value to check to divide data into two levels.
np.median(company["Sales"])
company["sales"]= "<=7.49"
company.loc[company["Sales"]>=7.49,"sales"]=">=7.49"

company["sales"].unique()
company["sales"].value_counts()


##Dropping Sales column from the data 
company.drop(["Sales"],axis=1,inplace = True)

## Company data has no null values
company.isnull().sum()
##There are no null values

##Checking the data type
company.info()

##As, the fit does not consider the String data, we need to encode the data.
from sklearn import preprocessing 
le = preprocessing.LabelEncoder()
for column_name in company.columns:
    if company[column_name].dtype == object:
        company[column_name] = le.fit_transform(company[column_name])
    else:
        pass

features = company.iloc[:,0:10] 
labels = company.iloc[:,10]
##Splitting the data
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.3,stratify = labels)

y_train.value_counts()
y_test.value_counts()

##Building the model
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

model = DT(criterion='entropy') 
model.fit(x_train,y_train)

##prediction on Training data
pred_train = pd.DataFrame(model.predict(x_train))

##Finding Accuracy for train data
acc_train = accuracy_score(y_train,pred_train)
##100%

## Confusion matrix
confusion_mat = pd.DataFrame(confusion_matrix(y_train,pred_train,))

##prediction on test data
pred_test = pd.DataFrame(model.predict(x_test))

##Accuracy on test data
acc_test = accuracy_score(y_test,pred_test)
##70%

##Confusion matrix
confusion_test = pd.DataFrame(confusion_matrix(y_test,pred_test))


##Visualization of decision trees
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus

colnames = list(company.columns)
predictors = colnames[:10]
target = colnames[10]

dot_data = StringIO()

export_graphviz(model,out_file = dot_data, filled =True, rounded = True, feature_names =predictors,class_names = target, impurity = False )
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

##Creating the pdf file of decision tree
graph.write_pdf('company.pdf')

##Creating a png file of the decsion tree
graph.write_png('company.png')

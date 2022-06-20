import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy as sp
import warnings
import datetime
warnings.filterwarnings("ignore")
data= pd.read_csv("fire-_m.csv")
data.head()
data.describe()
data.info()
data.value_counts
data.shape
data.columns
data.dtypes
data.isnull().sum()
data.isnull().any()
data.drop_duplicates(inplace = True)
#lets find the categorialfeatures
list_1=list(data.columns)
list_cate=[]
for i in list_1:
    if data[i].dtype=='object':
        list_cate.append(i)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in list_cate:
    data[i]=le.fit_transform(data[i])
data
data.hist(figsize=(20,14),color='g')
plt.show()
data.columns
sns.boxplot(x='year',y='month',data=data)
sns.stripplot(x='year',y='state',data=data)
sns.scatterplot(x='year',y='number',data=data,color='g')
sns.lineplot(x='year',y='number',data=data,color='g')
data.corr()
sns.heatmap(data.corr(), annot = True, cmap = 'viridis')
sns.pairplot(data=data)
sns.jointplot(x='year',y='month',data=data,color='r')
sns.regplot(x='year',y='number',data=data,color='b')
sns.kdeplot(x='year',y='state',data=data,color='g')
plt.style.use("default")
sns.barplot(x="year", y="number",data=data)
plt.title("YEAR vs NUMBER",fontsize=15)
plt.xlabel("YEAR")
plt.ylabel("NUMBER")
plt.show()
plt.style.use("default")
sns.barplot(x="month", y="state",data=data)
plt.title("MONTH vs STATE",fontsize=15)
plt.xlabel("MONTH")
plt.ylabel("STATE")
plt.show()
#lets find the categorialfeatures
list_1=list(data.columns)
list_cate=[]
for i in list_1:
    if data[i].dtype=='object':
        list_cate.append(i)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in list_cate:
    data[i]=le.fit_transform(data[i])
data
y=data['year']
x=data.drop('year',axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
print(len(x_test))
print(len(x_train))
print(len(y_test))
print(len(y_train))
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",knn.score(x_train,y_train)*100)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",gnb.score(x_train,y_train)*100)
print(accuracy_score(y_test,y_pred)*100)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=6, random_state=123,criterion='entropy')

dtree.fit(x_train,y_train)
from sklearn.metrics import classification_report,mean_squared_error
y_pred=dtree.predict(x_test)
conf =print(confusion_matrix(y_test, y_pred))
clf =print(classification_report(y_test, y_pred))
print("Training Score:\n",dtree.score(x_train,y_train)*100)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
conf =print(confusion_matrix(y_test, y_pred))
clf =print(classification_report(y_test, y_pred))
score=accuracy_score(y_test,y_pred)
score
print("Training Score:\n",rfc.score(x_train,y_train)*100)
from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier(base_estimator = None)
adb.fit(x_train,y_train)
y_pred=adb.predict(x_test)
conf =print(confusion_matrix(y_test, y_pred))
clf =print(classification_report(y_test, y_pred))
score=accuracy_score(y_test,y_pred)
score
print("Training Score:\n",adb.score(x_train,y_train)*100)
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(x_train,y_train)
y_pred=gbc.predict(x_test)
conf =print(confusion_matrix(y_test, y_pred))
clf =print(classification_report(y_test, y_pred))
score=accuracy_score(y_test,y_pred)
print(score)
print("Training Score:\n",gbc.score(x_train,y_train)*100)
data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
data
from xgboost import XGBClassifier

xgb =XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

xgb.fit(x_train, y_train)
y_pred=xgb.predict(x_test)
conf =print(confusion_matrix(y_test, y_pred))
clf =print(classification_report(y_test, y_pred))
score=accuracy_score(y_test,y_pred)
score
print("Training Score:\n",xgb.score(x_train,y_train)*100)
from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(n_estimators=100, random_state=0)
etc.fit(x_train,y_train)
y_pred=etc.predict(x_test)
conf =print(confusion_matrix(y_test, y_pred))
clf =print(classification_report(y_test, y_pred))
score=accuracy_score(y_test,y_pred)
print("Training Score:\n",etc.score(x_train,y_train)*100)
data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred+3})

print (data)

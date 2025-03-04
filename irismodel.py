import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import roc_curve,roc_auc_score

import warnings
warnings.filterwarnings("ignore")

# 1)Problem Statement
#Check in which species the flower fall i.e.setosa/virginica/versicolor

# 2)Data Gathering
df=pd.read_csv("Iris.csv")
df

df.info()

# 3)Exploratory Data Analysis(EDA)
df["Species"].value_counts()

df.nunique()

# 4)Feature engineering
#Remove Id column as it is having more unique values
df.drop("Id",axis=1,inplace=True)

# 5)Feature selection

sns.pairplot(df,hue="Species")

# 6)Model Training
# seperate dependent & independent variables

x=df.drop("Species",axis=1)
y=df["Species"]

y.value_counts()

# Split the model into training & testing

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42,stratify=y)

y_train.value_counts()

# Model building & instantiating Logistic regression
lr_clf = LogisticRegression(multi_class="ovr")
lr_clf.fit(x_train,y_train)


# 7)Model Evaluation

#calculate y predicted value
y_predict=lr_clf.predict(x_test)
y_predict

#Calculate y actual
y_test

# Accuracy
accuracy=accuracy_score(y_test,y_predict)
accuracy
#Confusion Matrix

cnf_matrix=confusion_matrix(y_test,y_predict)
cnf_matrix

#Multilabel Confusion Matrix

mlt_cnf_matrix=multilabel_confusion_matrix(y_test,y_predict)
mlt_cnf_matrix

# classification report

clf_report=classification_report(y_test,y_predict)
print("Classifiacation report is:\n",clf_report)

#Precision = TP/(TP+FP)
#precision for Iris-virginica 
Precision=12/(12+3)
Precision

#Recall = TP/(TP+FN)
#Recall for Iris-versicolor
Recall=10/(10+3)
Recall

# predict on single row
x_test

x_test.iloc[4]

SepalLengthCm   = 5.6
SepalWidthCm    = 2.0
PetalLengthCm   = 4.5
PetalWidthCm    = 1.5

test_array = np.array([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
test_array

column_names=x.columns
column_names

test_array=np.zeros(len(x.columns))
test_array

#define index of each column with the help of test array
test_array[0]=SepalLengthCm
test_array[1]=SepalWidthCm
test_array[2]=PetalLengthCm
test_array[3]=PetalWidthCm

test_array

#Predict model for different values

SepalLengthCm   = 4.6
SepalWidthCm    = 2.5
PetalLengthCm   = 3.5
PetalWidthCm    = 1.9

lr_clf.predict([test_array])[0]

#Now prepare pickle file of the model it is write inside our directory folder

import pickle
with open("Logistic_model.pkl","wb")as f:
    pickle.dump(lr_clf,f)

#here label encoded values are not present so we create json for column values only

project_data = {'columns' :list(x.columns)}

import json

with open("Project_data.json","w") as f:
    json.dump(project_data,f)
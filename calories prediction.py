# import the required dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt, seaborn as sns
import numpy as np


# loading the dataset
calories_dataset_input = pd.read_csv("exercise.csv").drop(columns=['User_ID'],axis=1)
calories_dataset_label = pd.read_csv("calories.csv").drop(columns=['User_ID'],axis=1)
calories_dataset = pd.concat([calories_dataset_input,calories_dataset_label],axis=1)
# show the first five rows in the dataset
calories_dataset.head()
# show the last five rows in the dataset
calories_dataset.tail()
# show dataset shape
calories_dataset.shape
# show some statistical info about the dataset
calories_dataset.describe()


# check about the none(missing) values in the dataset if will make a data cleaning or not
calories_dataset.isnull().sum()


# count the values of the Gender column and plot it
plt.figure(figsize=(5,5))
calories_dataset['Gender'].value_counts()
sns.countplot(x = 'Gender',data = calories_dataset) 


plt.figure(figsize=(10,10))
# plot the distribution of Calories column
plt.subplot(4,2,1)
sns.distplot(calories_dataset['Calories'],color='orange') 
# plot the distribution of Age column
plt.subplot(4,2,2)
sns.distplot(calories_dataset['Age'],color='red') 
# plot the distribution of Height column
plt.subplot(4,2,3)
sns.distplot(calories_dataset['Height'],color='blue') 
# plot the distribution of Weight column
plt.subplot(4,2,4)
sns.distplot(calories_dataset['Weight'],color='green') 
# plot the distribution of Duration column
plt.subplot(4,2,5)
sns.distplot(calories_dataset['Duration'],color='yellow') 
# plot the distribution of Heart_Rate column
plt.subplot(4,2,6)
sns.distplot(calories_dataset['Heart_Rate'],color='orange') 
# plot the distribution of Body_Temp column
plt.subplot(4,2,7)
sns.distplot(calories_dataset['Body_Temp'],color='green') 


# show the dataset before labeling
calories_dataset.head()
# label the Gender column by convert it from textual to numeric column
calories_dataset.replace({'Gender':{'male':0,'female':1}},inplace=True)
# show the dataset before labeling
calories_dataset.head()


# find the correlation between all the features in the dataset
correlation_values = calories_dataset.corr()
# determine the figure size of the plot
plt.figure(figsize=(10,10))
# plot the correlation of dataset
sns.heatmap(correlation_values,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')


# split the dataset into input and label data
X = calories_dataset.drop(columns=['Calories'])
Y = calories_dataset['Calories']
print(X)
print(Y)
# split the data into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,random_state=2)
print(X.shape,x_train.shape,x_test.shape)
print(Y.shape,y_train.shape,y_test.shape)


# Create the model and train it
XGBModel = XGBRegressor()
XGBModel.fit(x_train,y_train)
# Make the model predict the train and test input data
predicted_train_values = XGBModel.predict(x_train)
predicted_test_values = XGBModel.predict(x_test)
# avaluate the model accuracy
accuracy_train_prediction = r2_score(predicted_train_values,y_train)
accuracy_test_prediction = r2_score(predicted_test_values,y_test)
print(accuracy_train_prediction,accuracy_test_prediction)


# Making a predictive system
input_data = (0,61,179.0,84.0,27.0,116.0,40.7)
# convert input data into 1D numpy array
input_array = np.array(input_data)
# convert 1D input array into 2D
input_2D_array = input_array.reshape(1,-1)
# Make the model predict the output
print(XGBModel.predict(input_2D_array))

# Calories-Burnt-Prediction

This is a calories prediction model that uses the XGBoost regressor algorithm to predict the number of calories burned during exercise based on various features. 
The model has achieved an impressive accuracy of 998% on the test data.
Please note that an accuracy of 998% may indicate an issue with the model evaluation or that the dataset is not suitable for traditional regression problems.

## Prerequisites
Before running the code, make sure you have the following dependencies installed:
- Pandas
- XGBoost
- Scikit-learn
- Matplotlib
- Seaborn
- Numpy

## Overview of the Code
1-Load the calories dataset, which consists of input features and target labels. Concatenate the two dataframes.

2-Data Exploration:
- Check for missing values in the dataset.
- Visualize the count of gender in the dataset.
- Plot the distribution of various columns, including 'Calories', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', and 'Body_Temp'.

3-Data Preprocessing:
- Label encode the 'Gender' column to convert it from textual to numeric.

4-Explore the correlation between features using a heatmap.

5-Split the dataset into input features (X) and labels (Y). Remove the 'Calories' column.

6-Split the data into training and testing sets using a 70/30 split.

7-Create an XGBoost regressor model, train it on the training data, and predict on both the training and test data.

8-Evaluate the model's accuracy using the R-squared metric for both training and test data.

9-Create a predictive system to predict calorie burn for a new set of input data.


## Model Accuracy
The model has achieved an accuracy of 998% on the test data. 
However, this accuracy value appears to be unusually high and may indicate a problem with the model evaluation or dataset.

## Contribution
Contributions to this project are welcome. If you'd like to improve the accuracy, address any issues with the dataset or model evaluation, or enhance the data preprocessing and visualization steps, your contributions are appreciated. 
Please feel free to make any contributions and submit pull requests.


#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      mahal
#
# Created:     29/10/2024
# Copyright:   (c) mahal 2024
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot  as pit
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

data=pd.read_csv("housing.csv")
print(data.head())
print(data.shape)
print(data.describe)


#data cleaning

print(data.isnull().sum())

#data visualization
sns.relplot(x='price',y='area',data=data)
sns.relplot(x='price',y='stories',data=data)
pit.show()



#training and testing of the data



X=data.drop("price",axis=1)
Y=data["price"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5,random_state=123)
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

y_pred=regressor.predict(X_test)



#prediction
mse = mean_squared_error(y_pred,Y_test)
mae = mean_absolute_error(y_pred,Y_test)
print(mse,mae)
import math
rmse = math.sqrt(mse)
print(rmse)


pickle.dump(regressor, open("model.pkl", 'wb'))
model1 = pickle.load(open("model.pkl", 'rb'))

# Ensure the input for prediction has the correct feature names
input_features = pd.DataFrame([[23, 54, 12788, 3, 2, 1, 0, 1, 0, 1, 0, 1]], columns=X.columns)
print(model1.predict(input_features))
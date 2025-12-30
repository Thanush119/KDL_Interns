import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df=pd.read_csv("C:\\Users\\THANUSH\\Desktop\\Datasets\\IRIS.csv")


x=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm']]
y=df['PetalWidthCm']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#print('coefficient', model.coef_)
#print('intercept', model.intercept_)
#print('MSE:',mean_squared_error(y_test,y_pred))
#print("R2_Score:", r2_score(y_test,y_pred))
print("Actual values:", y_test.values)
print("Predicted values:", y_pred)
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd             
iris=pd.read_csv("C:\\Users\\THANUSH\\Desktop\\Datasets\\IRIS.csv")
print(iris.head(20))
print(iris.columns)
print(iris.dtypes)
print(iris.tail(10))
print(iris.shape)
print(iris.iloc[20:30,:])
plt.title("SEPAL LENGTH ANALYSIS")
plt.scatter(x=iris.index,y=iris['sepal length cm'],hue=iris["species"])
plt.show()
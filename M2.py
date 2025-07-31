import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np  
data_set=pd.read_csv("C:\\Users\\THANUSH\\Desktop\\PDFs\\temp_dataset.csv")
y=data_set.iloc[:,:].values
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(y)
y= imputer.transform(y)
print(y)
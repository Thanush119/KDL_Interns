import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

dt=pd.read_csv("C:\\Users\\THANUSH\\Desktop\\Datasets\\extern.csv")
print(dt.head())

# x=dt.iloc[:, :-1].values
# y=dt.iloc[:, 3].values
# x=dt[['Age'],['Abdominal_Pain'],['Diarrhea'],['Fever'],['Blood_in_Stool'],['Weight_Loss']]
# y=dt[['Diagnosis']]
# x=dt['Abdominal_Pain']
# y=dt['Diagnosis']
# features = dt.drop('Diagnosis', axis=1)
# y = dt['Diagnosis']
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(features)
# x = pd.DataFrame(X_scaled, columns=features.columns)


dt.plot(x='Hours_Studied', y='Passed', style='o')
plt.title('Dataset Visualization')
plt.xlabel('Hours_Studied')
plt.ylabel('Passed')
plt.show()

x=dt[['Hours_Studied']]
y=dt['Passed']
x_train,x_test,y_train, y_test= train_test_split(x,y,testsize=0.2,random_state=0)
lgr=LogisticRegression()
model=lgr.fit(x_train,y_train)
y_pred=lgr.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
print("Confusion Matrix")
print(cm)
print("Accuracy: ",acc)
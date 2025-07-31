import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('C:\\Users\\THANUSH\\Desktop\\lidp.csv')
label_encoder = LabelEncoder()  
data['Abdominal_Pain'] = label_encoder.fit_transform(data['Abdominal_Pain'])
data['Diarrhea'] = label_encoder.fit_transform(data['Diarrhea'])
data['Fever'] = label_encoder.fit_transform(data['Fever'])
data['Blood_in_Stool'] = label_encoder.fit_transform(data['Blood_in_Stool'])
data['Weight_Loss'] = label_encoder.fit_transform(data['Weight_Loss'])
data['Diagnosis'] = label_encoder.fit_transform(data['Diagnosis'])
X= data.drop('Diagnosis', axis=1)
y= data['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
print("Logistic Regression Report:\n", classification_report(y_test, log_pred))
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print("Decision Tree Report:\n", classification_report(y_test, dt_pred))
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest Report:\n", classification_report(y_test, rf_pred))
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Ulcer', 'Infection'], yticklabels=['Healthy', 'Ulcer', 'Infection'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
plot_confusion_matrix(y_test, log_pred, 'Logistic Regression')
plot_confusion_matrix(y_test, dt_pred, 'Decision Tree')
plot_confusion_matrix(y_test, rf_pred, 'Random Forest')
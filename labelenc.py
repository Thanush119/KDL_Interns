from sklearn.preprocessing import LabelEncoder
import pandas as pd     
colors=['red', 'blue', 'green', 'blue', 'red', 'green']
le=LabelEncoder()
encoded_colors=le.fit_transform(colors)
print("Original colors:", colors)  
print("Encoded colors:", encoded_colors)
print("Classes:", le.classes_)
def conv(n):
    rem=list()
    while n/2!=1:
        x=n%2
        rem.insert(x)
    rem.insert(1)
    print(rem)

conv(5)

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
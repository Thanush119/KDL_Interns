import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE


df = pd.read_csv('train_motion_data.csv')
df.columns = df.columns.str.strip()

plt.figure(figsize=(12, 6))
plt.plot(df['Timestamp'], df['AccX'], label='AccX', color='blue')
plt.plot(df['Timestamp'], df['AccY'], label='AccY', color='orange')
plt.plot(df['Timestamp'], df['AccZ'], label='AccZ', color='green')
plt.title('Accelerometer Raw Data: X, Y, Z Axes Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Acceleration (g or mg)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df['Timestamp'], df['GyroX'], label='GyroX', color='blue')
plt.plot(df['Timestamp'], df['GyroY'], label='GyroY', color='orange')
plt.plot(df['Timestamp'], df['GyroZ'], label='GyroZ', color='green')
plt.title('Gyroscope Raw Data: X, Y, Z Axes Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Angular Velocity (deg/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

df['Accel_Magnitude'] = np.sqrt(df['AccX']*2 + df['AccY']2 + df['AccZ']*2)
df['Gyro_Magnitude'] = np.sqrt(df['GyroX']*2 + df['GyroY']2 + df['GyroZ']*2)

accel_slow_thresh = 0.3         
accel_aggressive_thresh = 2.0   
gyro_slow_thresh = 50           
gyro_aggressive_thresh = 150 

def classify_behavior(row):
    a = row['Accel_Magnitude']
    g = row['Gyro_Magnitude']
    
    # Assign labels based on thresholds
    if a <= accel_slow_thresh and g <= gyro_slow_thresh:
        return 'Slow'
    elif a >= accel_aggressive_thresh or g >= gyro_aggressive_thresh:
        return 'Aggressive'
    else:
        return 'Normal'
        
df['Behavior_Category'] = df.apply(classify_behavior, axis=1)

# Filter out classes with fewer than 2 samples 
counts = df['Behavior_Category'].value_counts()
valid_classes = counts[counts > 1].index.tolist()
df = df[df['Behavior_Category'].isin(valid_classes)]

feature_cols = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
X = df[feature_cols]
y = df['Behavior_Category']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# ------------------------ Apply SMOTE to Balance Training Data ------------------------

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("Original class distribution:", np.bincount(y_train))
print("Balanced class distribution:", np.bincount(y_train_balanced))

# ------------------------ Train Base Models on Balanced Data ------------------------

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_balanced, y_train_balanced)
y_pred_lr = lr.predict(X_test)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))
print("LR Accuracy:", accuracy_score(y_test, y_pred_lr))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_balanced, y_train_balanced)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
print("RF Accuracy:", accuracy_score(y_test, y_pred_rf))

# ------------------------ Hybrid Model Training ------------------------

# Predict probabilities on balanced training set
train_lr_probs = lr.predict_proba(X_train_balanced)
train_rf_probs = rf.predict_proba(X_train_balanced)
train_hybrid = np.concatenate([train_lr_probs, train_rf_probs], axis=1)

# Predict probabilities on test set
test_lr_probs = lr.predict_proba(X_test)
test_rf_probs = rf.predict_proba(X_test)
test_hybrid = np.concatenate([test_lr_probs, test_rf_probs], axis=1)


# Train hybrid model (Random Forest)
final_clf = RandomForestClassifier(n_estimators=100, random_state=42)
final_clf.fit(train_hybrid, y_train_balanced)

# Predict and evaluate hybrid model
y_pred_final = final_clf.predict(test_hybrid)
print("\nFinal Hybrid Classifier Report (RFC):")
print(classification_report(y_test, y_pred_final, target_names=le.classes_))
print("Final Hybrid Classifier Accuracy:", accuracy_score(y_test, y_pred_final))

# ------------------------ Visualize Class Distribution (Optional) ------------------------

def plot_class_distribution(y_before, y_after, label_encoder):
    labels = label_encoder.inverse_transform(np.unique(y_after))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].bar(labels, np.bincount(y_before))
    axs[0].set_title('Before SMOTE')

    axs[1].bar(labels, np.bincount(y_after))
    axs[1].set_title('After SMOTE')

    for ax in axs:
        ax.set_ylabel("Count")
        ax.set_xlabel("Class")

    plt.suptitle("Class Distribution")
    plt.tight_layout()
    plt.show()

plot_class_distribution(y_train, y_train_balanced, le)

# ------------------------ Load and Predict on Separate Test Dataset ------------------------

test_df = pd.read_csv('test_motion_data.csv')
test_df.columns = test_df.columns.str.strip()

# Compute magnitudes
test_df['Accel_Magnitude'] = np.sqrt(test_df['AccX']*2 + test_df['AccY']2 + test_df['AccZ']*2)
test_df['Gyro_Magnitude'] = np.sqrt(test_df['GyroX']*2 + test_df['GyroY']2 + test_df['GyroZ']*2)

# Classify behavior
test_df['Behavior_Category'] = test_df.apply(classify_behavior, axis=1)

# Filter out unknown classes
mask = test_df['Behavior_Category'].isin(le.classes_)
test_df_filtered = test_df[mask]

# Predict on filtered test data
X_test_final = test_df_filtered[feature_cols]
y_test_final = le.transform(test_df_filtered['Behavior_Category'])

# Create hybrid test features
test_lr_probs = lr.predict_proba(X_test_final)
test_rf_probs = rf.predict_proba(X_test_final)
test_hybrid_features = np.concatenate([test_lr_probs, test_rf_probs], axis=1)

# Final predictions
y_pred_indices = final_clf.predict(test_hybrid_features)
y_pred_labels = le.inverse_transform(y_pred_indices)

test_df_filtered['Predicted_Label'] = y_pred_labels

# Show predictions
print("\nTest Predictions (First 20 rows):")
print(test_df_filtered[['Behavior_Category', 'Predicted_Label']].head(20))
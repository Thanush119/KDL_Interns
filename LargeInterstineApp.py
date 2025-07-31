import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import norm
import streamlit as st

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv("C:\\Users\\THANUSH\\Desktop\\large_intestine_disease_prediction(AutoRecovered).csv")
    return data

# Main function
def main():
    st.title("Large Intestine Disease Prediction System")
    st.write("This application analyzes patient data to predict the likelihood of large intestine disease.")
    
    # Load data
    data = load_data()
    st.write("Dataset loaded successfully.")
    
    # Display data overview
    if st.checkbox("Show Data Overview"):
        st.write(data.info())
        st.write(data.describe())
        st.write(data.isnull().sum())
    
    # Encode categorical columns
    cat_cols = [col for col in data.columns if data[col].dtype == 'object']
    if cat_cols:
        le = LabelEncoder()
        for col in cat_cols:
            data[col] = le.fit_transform(data[col])
    
    # Display correlation heatmap
    if st.checkbox("Show Correlation Heatmap"):
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
        plt.title("Correlation Heatmap of Large Intestine Disease Factors")
        st.pyplot(plt)

    # Prepare data for modeling
    features = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X = pd.DataFrame(X_scaled, columns=features.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train Logistic Regression model
    if st.button("Train Logistic Regression Model"):
        log_model = LogisticRegression(max_iter=200, random_state=42)
        log_model.fit(X_train, y_train)
        y_pred_log = log_model.predict(X_test)
        
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred_log))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred_log))
        st.write(f"Accuracy Score: {accuracy_score(y_test, y_pred_log):.4f}")

    # Naive Bayes Model
    classes = y.unique()
    class_stats = {}
    for c in classes:
        class_data = X[y == c]
        class_stats[c] = {
            'mean': class_data.mean(),
            'std': class_data.std(),
            'prior': len(class_data) / len(X)
        }

    # Patient input for prediction
    st.subheader("Interactive Recovery Probability Estimation")
    symptom_features = ['Abdominal_Pain', 'Diarrhea', 'Fever', 'Blood_in_Stool', 'Weight_Loss']
    patient_number = st.number_input("Enter a patient number (0 to {}):".format(len(X)-1), min_value=0, max_value=len(X)-1, value=0)

    if st.button("Predict for Patient"):
        new_patient = X.iloc[patient_number]
        st.write(f"--- Patient {patient_number} Symptom Profile ---")
        st.write(new_patient)

        has_zero_signs = all(abs(new_patient[f]) < 0.1 for f in symptom_features)

        if has_zero_signs:
            st.write("Patient shows virtually zero signs of disease. Recommendation: NO TREATMENT REQUIRED, JUST MEDICATION.")
        else:
            posteriors = {}
            for c in classes:
                prior = class_stats[c]['prior']
                likelihood = 1.0
                for f in features.columns:
                    mean = class_stats[c]['mean'][f]
                    std = class_stats[c]['std'][f]
                    x = new_patient[f]
                    std = std if std > 1e-6 else 1e-6
                    prob = norm.pdf(x, mean, std)
                    likelihood *= prob
                posteriors[c] = prior * likelihood

            total_posterior = sum(posteriors.values())
            if total_posterior == 0:
                for c in posteriors:
                    posteriors[c] = 1 / len(classes)
            else:
                for c in posteriors:
                    posteriors[c] /= total_posterior

            prediction = max(posteriors, key=posteriors.get)
            diagnosis_map = {0: "No Disease (Recovery)", 1: "Disease (No Recovery)"}
            naive_recovery_prob = posteriors.get(0, 0)
            naive_disease_prob = posteriors.get(1, 0)

            # Blended Recovery Estimation
            low_severity_count = sum(abs(new_patient[f]) < 0.5 for f in symptom_features)
            severity_score = low_severity_count / len(symptom_features)
            blended_recovery_prob = 0.7 * naive_recovery_prob + 0.3 * severity_score

            st.write(f"--- Naive Bayes Prediction for Patient {patient_number} ---")
            st.write(f"Predicted Class: {diagnosis_map.get(prediction)}")
            st.write(f"Probability of Recovery (Naive Bayes): {naive_recovery_prob:.2f}")
            st.write(f"Recovery Likelihood Based on Symptom Thresholds: {severity_score:.2f}")
            st.write(f"Final Blended Recovery Probability: {blended_recovery_prob:.2f}")

            if blended_recovery_prob >= 0.7:
                st.write("High recovery probability. Mild treatment or medication advised.")
            elif 0.4 <= blended_recovery_prob < 0.7:
                st.write("Moderate recovery chance. Recommend diagnostic tests.")
            else:
                st.write("Low recovery probability. Immediate medical attention required.")

    st.write("--- End of Program ---")
    st.write("Thank you for using the Large Intestine Disease Prediction System.")

if __name__ == "__main__":
    main()

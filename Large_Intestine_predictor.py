import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import norm

print("Welcome to the Large Intestine Disease Prediction System!")
print("This script will analyze patient data to predict the likelihood of large intestine disease.")
print("We'll use two machine learning approaches: Logistic Regression and a custom Naive Bayes classifier.")
print("-" * 70)

try:
    data = pd.read_csv("C:\\Users\\THANUSH\\Desktop\\large_intestine_disease_prediction(AutoRecovered).csv")
    print("Dataset 'large_intestine_disease_prediction(AutoRecovered).csv' loaded successfully.")
except FileNotFoundError:
    print("Error: The dataset file was not found.")
    exit()

print("\n--- Initial Data Overview ---")
print(data.info())
print(data.describe())
print(data.isnull().sum())

cat_cols = [col for col in data.columns if data[col].dtype == 'object']
if cat_cols:
    le = LabelEncoder()
    print(f"\nFound categorical columns: {', '.join(cat_cols)}. Encoding them now...")
    for col in cat_cols:
        data[col] = le.fit_transform(data[col])
    print("Categorical columns encoded successfully.")
else:
    print("No categorical columns found.")

plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
plt.title("Correlation Heatmap of Large Intestine Disease Factors", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

features = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
X = pd.DataFrame(X_scaled, columns=features.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("\n--- Training Logistic Regression ---")
log_model = LogisticRegression(max_iter=200, random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
print(f"Accuracy Score: {accuracy_score(y_test, y_pred_log):.4f}")

print("\n--- Severe Cases from Logistic Regression ---")
severity_criteria = (
    (X_test['Abdominal_Pain'] > 0.8) |
    (X_test['Fever'] > 0.8) |
    (X_test['Blood_in_Stool'] > 0.8) |
    (X_test['Diarrhea'] > 0.8) |
    (X_test['Weight_Loss'] > 0.8)
)
severe_patients_indices = X_test[severity_criteria].index.tolist()
predicted_disease_patients_indices = X_test[y_pred_log == 1].index.tolist()
print("Patients with severe symptoms:", severe_patients_indices)
print("Patients predicted to have disease:", predicted_disease_patients_indices)

print("\n--- Naive Bayes Model ---")
X_nb = X.copy()
y_nb = y.copy()
classes = y_nb.unique()
class_stats = {}

for c in classes:
    class_data = X_nb[y_nb == c]
    class_stats[c] = {
        'mean': class_data.mean(),
        'std': class_data.std(),
        'prior': len(class_data) / len(X_nb)
    }


print("\n--- Interactive Recovery Probability Estimation ---")
symptom_features = ['Abdominal_Pain', 'Diarrhea', 'Fever', 'Blood_in_Stool', 'Weight_Loss']
ZERO_SIGNS_THRESHOLD = 0.1
LOW_SEVERITY_THRESHOLD = 0.5  

while True:
    try:
        patient_number_input = input(f"\nEnter a patient number (0 to {len(X_nb)-1}, or -1 to exit): ")
        patient_number = int(patient_number_input)

        if patient_number == -1:
            print("Exiting interactive mode.")
            break

        if 0 <= patient_number < len(X_nb):
            new_patient = X_nb.iloc[patient_number]
            print(f"\n--- Patient {patient_number} Symptom Profile ---")
            print(new_patient)

            has_zero_signs = all(abs(new_patient[f]) < ZERO_SIGNS_THRESHOLD for f in symptom_features)

            if has_zero_signs:
                print("\n Patient shows virtually zero signs of disease.")
                print("Recommendation: NO TREATMENT REQUIRED, JUST MEDICATION.")
                continue

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

            # --- Blended Recovery Estimation ---
            low_severity_count = sum(abs(new_patient[f]) < LOW_SEVERITY_THRESHOLD for f in symptom_features)
            severity_score = low_severity_count / len(symptom_features)
            blended_recovery_prob = 0.7 * naive_recovery_prob + 0.3 * severity_score

            print(f"\n--- Naive Bayes Prediction for Patient {patient_number} ---")
            print(f"Predicted Class: {diagnosis_map.get(prediction)}")
            print(f"Probability of Recovery (Naive Bayes): {naive_recovery_prob:.2f}")
            print(f"Recovery Likelihood Based on Symptom Thresholds: {severity_score:.2f}")
            print(f"Final Blended Recovery Probability: {blended_recovery_prob:.2f}")

            if blended_recovery_prob >= 0.7:
                print(" High recovery probability. Mild treatment or medication advised.")
            elif 0.4 <= blended_recovery_prob < 0.7:
                print(" Moderate recovery chance. Recommend diagnostic tests.")
            else:
                print(" Low recovery probability. Immediate medical attention required.")

        else:
            print("Invalid patient number.")
    except ValueError:
        print("Please enter a valid integer.")
    except Exception as e:
        print(f"Error during prediction: {e}")

print("\n--- End of Program ---")
print("Thank you for using the Large Intestine Disease Prediction System.")

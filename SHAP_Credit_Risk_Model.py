import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from  SHAP_Framework_Implementations import *


#Load data
data = pd.read_csv("credit_risk_dataset.csv")
data = data.dropna()

#Features names
features_names = ['Age', 'Income', 'Home Ownership', 'Employment Length', 'Loan Intent', 'Loan Grade', 'Loan Amount', 'Interest Rate', 'Percent Income', 'Historical Default', 'Credit History Length']
numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

#Encode categorical features
le = LabelEncoder()
for i in categorical_features:
  data[i] = le.fit_transform(data[i])

#Variance analysis
print("Variance Analysis")
print(data.var())
print("")

import numpy as np

#High variance features treatment

high_var_features = ["person_age", "person_income", "person_emp_length",  "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"]
for i in high_var_features:
  data[i] = np.log(1 + data[i])

print("New Variances")
print(data.var())
print("")

#Extract features and target
X = data.drop('loan_status', axis=1)
y = data['loan_status'].values

print(X.head())
print("")



from sklearn.model_selection import train_test_split
#Make train and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#Standarize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_prepared = scaler.fit_transform(X_train)
X_test_prepared = scaler.transform(X_test)

print(pd.DataFrame(scaler.transform(X)).head())
print("")


import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

keras.utils.set_random_seed(42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_prepared.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train_prepared, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

y_pred_prob = model.predict(X_test_prepared)
y_pred = (y_pred_prob > 0.5).astype('int')

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, precision_score, classification_report

y_pred_prob = model.predict(X_test_prepared)
y_pred = (y_pred_prob > 0.5).astype('int')

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print(f'Recall (Sensibilidad): {recall:.2f}')
print(f'Precision: {precision:.2f}')

print(classification_report(y_test, y_pred))


x0 = np.zeros(model.input_shape[1])
SHAP_sample = X_test_prepared[0:200]
Deep_SHAP_values = get_Deep_SHAP_Values(model, SHAP_sample, x0)
SHAP_plot(SHAP_sample, Deep_SHAP_values, "Deep SHAP Values", feature_names = features_names)

#Execution Times

# import time

# inicio = time.time()

# Kernel_SHAP_values = get_Kernel_SHAP_values(model, SHAP_sample, SHAP_sample, x0)

# fin = time.time()
# tiempo_total = fin - inicio
# print("The execution time is:", tiempo_total, "seconds")


# import time

# inicio2 = time.time()

# SHAP_values = get_SHAP_values(model, SHAP_sample, SHAP_sample, x0)

# fin2 = time.time()
# tiempo_total2 = fin2 - inicio2
# print("The execution time is:", tiempo_total2, "seconds")


# import time

# inicio3 = time.time()

# Deep_SHAP_values = get_Deep_SHAP_Values(model, SHAP_sample, x0)

# fin3 = time.time()
# tiempo_total3 = fin3 - inicio3
# print("The execution time is:", tiempo_total3, "seconds")



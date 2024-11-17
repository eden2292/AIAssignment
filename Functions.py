import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def accuracy(y_test, y_pred, value_count):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Number of positives and negatives: \n {value_count}")

    print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred)}")
    print(f"Accuracy: {accuracy} \n"
          f"Precision: {precision} \n"
          f"Recall: {recall} \n"
          f"F1 Score: {f1}")
    
def predict_outcome(dataset, trained_classifier):
    ivd_outcome = []
    print(f'Enter prediction variables:')
    for i, column in enumerate(dataset.columns[:-1]):
        value = float(input(f"{column}"))
        ivd_outcome.append(value)

    ivd_outcome_df = pd.DataFrame([ivd_outcome], columns=dataset.columns[:-1])

    crash_prediction = trained_classifier.predict(ivd_outcome_df)
    if crash_prediction == 1:
        outcome = 'Crash'
    else:
        outcome = 'No crash'

    return outcome
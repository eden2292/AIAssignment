from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('data_multivar_nb.txt', header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nPrecision:", precision)
print("\nRecall:", recall)
print("\nF1 score:", f1)
print("\nConfusion Matrix:\n", confusion)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

x_pred = nb_classifier.predict(X_test)

nb_accuracy = accuracy_score(y_test, x_pred)
nb_precision = precision_score(y_test, x_pred, average='weighted')
nb_recall = recall_score(y_test, x_pred, average='weighted')
nb_f1 = f1_score(y_test, x_pred, average='weighted')
nb_confusion = confusion_matrix(y_test, x_pred)

print("\nIndicators of the naive Bayesian classifier")
print("\nAccuracy:", nb_accuracy)
print("\nPrecision:", nb_precision)
print("\nRecall:", nb_recall)
print("\nF1 score:", nb_f1)
print("\nConfusion Matrix:\n", nb_confusion)

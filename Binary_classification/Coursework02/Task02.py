from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from LoadFunction_sup import load_data
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import tree
from sklearn import svm
import numpy as np

# print(training_x)
# print(testing_x)
# print(train_labels)
# print(test_labels)

# SVM classifier
classifier_SVM = SVC()
dimensionality = [i for i in range(1, 16)]
accuracies_svm = []
accuracies_nn = []
accuracies_knn = []

# SVM classifier
for i in dimensionality:
    training_x, testing_x, train_labels, test_labels = load_data(i)

    classifier_SVM.fit(training_x, train_labels)
    results_SVM = classifier_SVM.predict(testing_x)
    precision_SVM = accuracy_score(test_labels, results_SVM)
    accuracies_svm.append(precision_SVM)

print('Accuracy of SVM classifier:', accuracies_svm)

# plt.plot(dimensionality, accuracies, linewidth=1, marker='o', markerfacecolor='white')
# plt.ylabel('Accuracy')
# plt.xlabel('Dimensionality of principle components')
# plt.xticks(flat)
# plt.grid(alpha=0.5)
# plt.show()


# NN classifier
for i in dimensionality:
    training_x, testing_x, train_labels, test_labels = load_data(i)

    classifier_NN = MLPClassifier(solver='adam', alpha=1e-5, activation='logistic',
                                  hidden_layer_sizes=(10, 6, 4, 2), max_iter=5000, random_state=1)
    classifier_NN.fit(training_x, train_labels)
    results_NN = classifier_NN.predict(testing_x)

    error_NN = ((results_NN - test_labels) ** 2).cumsum()[-1]
    precision_NN = 1 - error_NN / len(test_labels)

    accuracies_nn.append(precision_NN)

print('Accuracy of NN classifier:', accuracies_nn)

# KNN classifier
for i in dimensionality:
    training_x, testing_x, train_labels, test_labels = load_data(i)
    classifier_KNN = KNeighborsClassifier()
    classifier_KNN.fit(training_x, train_labels)

    results_KNN = classifier_KNN.predict(testing_x)
    precision_KNN = accuracy_score(test_labels, results_KNN)

    accuracies_knn.append(precision_KNN)

print('Accuracy of KNN classifier:', accuracies_knn)

fig, ax = plt.subplots()

ax.plot(dimensionality, accuracies_svm, linewidth=1, marker='o', markerfacecolor='white', label='SVM')
ax.plot(dimensionality, accuracies_nn, linewidth=1, marker='o', markerfacecolor='white', label='Neural Net')
ax.plot(dimensionality, accuracies_knn, linewidth=1, marker='o', markerfacecolor='white', label='KNN')

ax.set_xticks(dimensionality)
ax.grid(alpha=0.5)
ax.set_xlabel('Dimensionality of principle components')
ax.set_ylabel('Accuracy')
ax.legend()

plt.show()


# # Logistic classifier
# classifier_los = LogisticRegression(penalty='l1', C=1e3, solver='liblinear')
# classifier_los.fit(training_x, train_labels)
#
# results_los = classifier_los.predict(testing_x)
#
# precision_los = accuracy_score(results_los, test_labels)
#
# print('The precision of LogisticRegression is: ', precision_los)

# # # DT classifier
# classifier_DT = tree.DecisionTreeClassifier(class_weight='balanced', max_depth=6)
# classifier_DT.fit(training_x, train_labels)
#
# results_DT = classifier_DT.predict(testing_x)
#
# precision_DT = accuracy_score(test_labels, results_DT)
# print('The precision of DT classifier is: ', precision_DT)
#
#
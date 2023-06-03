from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from LoadFunction_sup import load_data
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# training_x, testing_x, train_labels, test_labels = load_data(10)
# parameters = {'kernel': ['poly', 'rbf'], 'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
# svc = SVC()
#
# clf = GridSearchCV(svc, parameters)
#
# clf.fit(training_x, train_labels)
#
# print(clf.cv_results_)
#
# score = clf.cv_results_['mean_test_score']
# C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#
# score_poly = []
# score_rbf = []
#
# for i in list(range(1, 21, 1)):
#     if i % 2 == 0:
#         score_poly.append(score[i-1])
#     else:
#         score_rbf.append(score[i-1])
#
# fig, ax = plt.subplots()
#
# ax.plot(C, score_rbf, linewidth=1, marker='o', markerfacecolor='white', label='RBF')
# ax.plot(C, score_poly, linewidth=1, marker='o', markerfacecolor='white', label='poly')
#
# ax.set_xticks(C)
# ax.grid(alpha=0.5)
# ax.set_xlabel('C')
# ax.set_ylabel('Score')
# ax.legend()
#
# plt.show()


# training_x, testing_x, train_labels, test_labels = load_data(11)
#
# parameters = {'activation': ['relu', 'logistic', 'tanh'], 'alpha': [0.01, 0.001, 0.0001, 0.00001]}
#
# nn = MLPClassifier(hidden_layer_sizes=(10, 6, 4, 2), max_iter=5000)
#
# clf = GridSearchCV(nn, parameters)
# clf.fit(training_x, train_labels)
#
# score = clf.cv_results_['mean_test_score']
#
# alpha = [0.01, 0.001, 0.0001, 0.00001]
#
# score_relu = score[:4]
# score_log = score[4:8]
# score_tanh = score[8:]
#
# fig, ax = plt.subplots()
#
# ax.plot(alpha, score_relu, linewidth=1, marker='o', markerfacecolor='white', label='Relu')
# ax.plot(alpha, score_log, linewidth=1, marker='o', markerfacecolor='white', label='Logistic')
# ax.plot(alpha, score_tanh, linewidth=1, marker='o', markerfacecolor='white', label='Tanh')
# ax.grid(alpha=0.5)
# ax.set_xlabel('C')
# ax.set_ylabel('Score')
# ax.legend()
#
# plt.show()


training_x, testing_x, train_labels, test_labels = load_data(10)

parameters = {'p': [1, 2], 'n_neighbors': list(range(1, 16, 2))}

knn = KNeighborsClassifier()
clf = GridSearchCV(knn, parameters)
clf.fit(training_x, train_labels)

print(clf.cv_results_)

score = clf.cv_results_['mean_test_score']
score_1 = []
score_2 = []


for i in list(range(1, 17, 1)):
    if i % 2 == 0:
        score_2.append(score[i-1])
    else:
        score_1.append(score[i-1])

fig, ax = plt.subplots()

ax.plot(list(range(1, 16, 2)), score_1, linewidth=1, marker='o', markerfacecolor='white', label='L1 distance')
ax.plot(list(range(1, 16, 2)), score_2, linewidth=1, marker='o', markerfacecolor='white', label='L2 distance')
ax.grid(alpha=0.5)
ax.set_xticks(list(range(1, 16, 2)))
ax.set_xlabel('K neighbors')
ax.set_ylabel('Score')
ax.legend()

plt.show()

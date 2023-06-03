from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from LoadFunction_sup import load_data
import matplotlib.pyplot as plt
from sklearn.svm import SVC

training_x_10, testing_x_10, train_labels_10, test_labels_10 = load_data(10)
training_x_11, testing_x_11, train_labels_11, test_labels_11 = load_data(11)

svc = SVC()
svc.fit(training_x_10, train_labels_10)
svc_result = svc.predict(testing_x_10)

cm_svc = confusion_matrix(test_labels_10, svc_result)
disp_svc = ConfusionMatrixDisplay(cm_svc)

disp_svc.plot()
plt.show()

print(f1_score(test_labels_10, svc_result))


# nn = MLPClassifier(hidden_layer_sizes=(10, 6, 4, 2), max_iter=5000)
# nn.fit(training_x_11, train_labels_11)
# nn_result = nn.predict(testing_x_11)
#
# cm_nn = confusion_matrix(test_labels_11, nn_result)
# disp_nn = ConfusionMatrixDisplay(cm_nn)
#
# disp_nn.plot()
# plt.show()
#
# print(f1_score(test_labels_11, nn_result))

# knn = KNeighborsClassifier(n_neighbors=15)
# knn.fit(training_x_10, train_labels_10)
# knn_result = knn.predict(testing_x_10)
#
# cm_knn = confusion_matrix(test_labels_10, knn_result)
# disp_knn = ConfusionMatrixDisplay(cm_knn)
#
# disp_knn.plot()
# plt.show()
#
# print(f1_score(test_labels_10, knn_result))




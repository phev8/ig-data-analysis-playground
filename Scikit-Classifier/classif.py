import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

from Skeleton import Global_Constants

main_dir = Global_Constants.main_dir
git_dir = Global_Constants.git_dir
project_dir = Global_Constants.project_dir
data_dir = Global_Constants.data_dir
filename = Global_Constants.filename
output_dir = Global_Constants.output_dir
skeletons_dir = Global_Constants.skeletons_dir

def read_feature_matrices_from_pickle_file():
    # global data_dir
    # set_paths()
    file = data_dir + 'Final_Feature_Matrix.pkl'

    pkl_file = open(file, "rb")
    matrix = pickle.load(pkl_file)
    return matrix

def create_train_matrix():
    feature_matrix = read_feature_matrices_from_pickle_file()

    # print(feature_matrix)

    input_matrix = [elem[1]+elem[2]+elem[3] for elem in feature_matrix]
    output_matrix = [elem[-1] for elem in feature_matrix]

    # print(input_matrix)
    # print(output_matrix)

    # input_matrix = [[1,2,3],[2,4,6],[3,6,9]]
    # output_matrix = ['1','2','3']

    x_train, x_test, y_train, y_test = train_test_split(input_matrix, output_matrix, test_size=0.3, random_state=1338)
    return [x_train, x_test, y_train, y_test]

def train_KNN():
    x_train, x_test, y_train, y_test = create_train_matrix()

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)

    pkl_file = open(data_dir + 'KNN_Classifier.pkl', 'wb')
    pickle.dump(neigh, pkl_file)
    print("Save Model")

def test_KNN():
    x_train, x_test, y_train, y_test = create_train_matrix()

    pkl_file = open(data_dir + 'KNN_Classifier.pkl', "rb")

    neigh = pickle.load(pkl_file)
    print(neigh.score(x_test, y_test))

def train_and_test(classifier_id):
    feature_matrix = read_feature_matrices_from_pickle_file()
    input_matrix = [elem[1]+elem[2]+elem[3] for elem in feature_matrix]
    output_matrix = [elem[-1] for elem in feature_matrix]
    x_train, x_test, y_train, y_test = create_train_matrix()

    if classifier_id == 0:
        classifier = KNeighborsClassifier(n_neighbors=3)
        name = 'KNN'
    if classifier_id == 1:
        classifier = GaussianNB()
        name = 'GaussianNB'
    if classifier_id == 2:
        classifier = MultinomialNB()
        name = 'MultinomialNB'
    if classifier_id == 3:
        classifier = BernoulliNB()
        name = 'BernoulliNB'
    if classifier_id == 4:
        classifier = svm.SVC(gamma='scale')
        name = 'SVC'
    if classifier_id == 5:
        classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        name = 'RandomForestClassifier'
    if classifier_id == 6:
        classifier = tree.DecisionTreeClassifier()
        name = 'DecisionTreeClassifier'

    classifier.fit(x_train, y_train)
    scores = cross_val_score(classifier, input_matrix, output_matrix, cv=3)
    predicted = classifier.predict(x_test)

    print(name,'results:')
    print('Accuracy score:', accuracy_score(y_test, predicted))
    print('Precision score:', precision_score(y_test, predicted, average='macro'))
    print('Recall score:', recall_score(y_test, predicted, average='macro'))
    print('f1 score:', f1_score(y_test, predicted, average='macro'))

    print('Cross Validation Scores:', scores)
    print("Mean Accuracy (Std): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

    print('confusion matrix:\n', confusion_matrix(y_test, predicted))

    print('\n')

if __name__ == '__main__':

    for i in range(7):
        classifier_id = i
        train_and_test(classifier_id)

"""
# print(create_train_matrix())
train_KNN()
print("Test Model")
test_KNN()
"""


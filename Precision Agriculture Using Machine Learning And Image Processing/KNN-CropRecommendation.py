import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize


def data_input():
    # your data code goes here
    dataset = pd.read_csv('Crop_recommendation.csv')
    newsubset = []
    dataset.drop(['rainfall'], axis=1, inplace=True)
    datalabels = ['cotton', 'maize', 'chickpea', 'kidneybeans']

    for i in datalabels:
        newsubset.append(dataset.loc[dataset['label'] == i])

    dataset_modified = pd.concat(newsubset, ignore_index = True)

    # dividing datset into dependant and independant variables
    X = dataset_modified.iloc[:, :-1].values
    y = dataset_modified.iloc[:, -1].values

    # normalization on set
    # X_norm = normalize(X,axis = 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=0)

    return X_train, X_test, y_train, y_test, X_val, y_val

def node_process():

    X_train, X_test, y_train, y_test, X_val, y_val = data_input()

    # applying feature scaling
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_val_scaled = sc.transform(X_val)
    X_test_scaled = sc.transform(X_test)

    # applying PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    color_train = []
    for i in y_train:
        if i == 'cotton':
            color_train.append('red')
        elif i == 'maize':
            color_train.append('blue')
        elif i == 'kidneybeans':
            color_train.append('purple')
        else:
            color_train.append('yellow')
    color_val = []
    for i in y_val:
        if i == 'cotton':
            color_val.append('black')
        elif i == 'maize':
            color_val.append('green')
        elif i == 'kidneybeans':
            color_val.append('orange')
        else:
            color_val.append('brown')

    red_patch = mpatches.Patch(color='red', label='Cotton')
    purple_patch = mpatches.Patch(color='purple', label='Kidney beans')
    blue_patch = mpatches.Patch(color='blue', label='Maize')
    yellow_patch = mpatches.Patch(color='yellow', label='Chickpea')

    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=color_train)
    plt.scatter(X_val_pca[:, 0], X_val_pca[:, 1], c=color_val)

    plt.legend(handles=[red_patch, purple_patch, blue_patch, yellow_patch])
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    plt.title('Principal Component Analysis')
    plt.show()

    kVals = range(1, 30, 2)
    accuracies = []

    # loop over various values of `k` for the k-Nearest Neighbor classifier
    for k in range(1, 30, 2):
        # train the k-Nearest Neighbor classifier with the current value of `k`
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        # evaluate the model and update the accuracies list
        score = model.score(X_val, y_val)
        # print("k=%d, accuracy=%.2f%%" % (k, score * 100))
        accuracies.append(score)

    # find the value of k that has the largest accuracy
    i = int(np.argmax(accuracies))
    # print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],accuracies[i] * 100))

    model = KNeighborsClassifier(n_neighbors=kVals[i])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # show a final classification report demonstrating the accuracy of the classifier
    # for each of the digits
    print("EVALUATION ON TESTING DATA FOR KNN\n")
    # ev = pd.DataFrame({'Original values': y_test, 'Predicted values': predictions})
    # print(ev)
    knn_score = model.score(X_test, y_test)
    print(classification_report(y_test, predictions))
    print(f'KNN model accuracy score : {knn_score*100}\n')

    # Implementation of Decision Trees Algorithm
    dt_clf = tree.DecisionTreeClassifier(splitter='random',max_depth=3)
    dt_clf = dt_clf.fit(X_train, y_train)
    dt_score = dt_clf.score(X_test, y_test)
    print("EVALUATION ON TESTING DATA FOR DECISION TREES\n")
    dt_predictions = dt_clf.predict(X_test)
    print(classification_report(y_test, dt_predictions))
    print(f'Decision Tree model accuracy score : {dt_score*100}\n')

    # Implementation of Support Vector Machine
    svm_clf = SVC(C=1, kernel='linear')
    svm_clf = svm_clf.fit(X_train, y_train)
    svm_score = svm_clf.score(X_test, y_test)
    print("EVALUATION ON TESTING DATA FOR SVM\n")
    svm_predictions = svm_clf.predict(X_test)
    print(classification_report(y_test, svm_predictions))
    print(f'SVM model accuracy score : {svm_score*100}\n')


    # return pca.n_components
'''
def transmission_fn(some_data):
    # take the data and transmit it through the channel
    # perhaps add some noise to the processed data
    return perturb(some_data)


def base_station_fn(collected_data):
    # collect the data and do some basic processing
    print(collected_data)
'''
node_process()

#if __name__ === "__main__":

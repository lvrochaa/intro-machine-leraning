def classify_svm(features_train, labels_train):
    from sklearn.svm import SVC
    clf = SVC(kernel="linear")
    clf.fit(features_train, labels_train)
    return clf


def svm_accuracy(clf, features_test, labels_test):
    from sklearn.metrics import accuracy_score
    pred = clf.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    return accuracy

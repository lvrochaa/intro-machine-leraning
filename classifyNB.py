def classify_nb(features_train, labels_train):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf


def nb_accuracy(clf, features_test, labels_test):
    from sklearn.metrics import accuracy_score
    pred = clf.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    return accuracy

def classify_dt(features_train, labels_train):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features_train, labels_train)
    return clf


def dt_accuracy(clf, features_test, labels_test):
    from sklearn.metrics import accuracy_score
    pred = clf.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    return accuracy

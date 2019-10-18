#!/usr/bin/python

""" Complete the code in ClassifyNB.py with the sklearn
    Naive Bayes classifier to classify the terrain data.

    The objective of this exercise is to recreate the decision
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """

from prepTerrainData import make_terrain_data
from classifyView import pretty_picture, output_image
import classifyNB as nb
import classifySVM as svm

features_train, labels_train, features_test, labels_test = make_terrain_data()
# grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
# bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
# grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
# bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

clfNB = nb.classify_nb(features_train, labels_train)
acuryNB = nb.nb_accuracy(clfNB, features_test, labels_test)

clfSVM = svm.classify_svm(features_train, labels_train)
acurySVM = svm.svm_accuracy(clfSVM, features_test, labels_test)

filename = "graphicNB"
pretty_picture(clfNB, features_test, labels_test, filename)
output_image("graficoNB.png", "png", open(filename+".png", "rb").read())
print("Taxa de acerto de Navie Bayes: {}".format(acuryNB))

filename = "graphicSVM"
pretty_picture(clfSVM, features_test, labels_test, filename)
output_image(filename+".png", "png", open(filename+".png", "rb").read())
print("Taxa de acerto do SVM: {}".format(acurySVM))

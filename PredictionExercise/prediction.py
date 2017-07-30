#!/usr/bin/env python

from classifier import GNB
from sklearn import tree

import json


def main():
    gnb = GNB()
    with open('train.json', 'r') as f:
        j = json.load(f)
    # print(j.keys())
    X = j['states']
    Y = j['labels']
    for i in range(len(X)):
        X[i][1] = X[i][1]%4
    gnb.train(X, Y)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X,Y)

    with open('test.json', 'r') as f:
        j = json.load(f)

    X = j['states']
    Y = j['labels']
    for i in range(len(X)):
        X[i][1] = X[i][1]%4
    score = 0

    # print(gnb.summaries)

    for coords, label in zip(X,Y):
        predicted = gnb.predict(coords)
        if predicted == label:
            score += 1
    fraction_correct = float(score) / len(X)
    print("Using GNB, I got {:%} correct".format(fraction_correct))

    p = clf.predict(X)
    score = 0
    for i in range(100):
        if p[i]==Y[i]:
            score += 1
    print("Using Decision Tree Classifier, I got {:%} correct".format(score/100.0))

if __name__ == "__main__":
    main()

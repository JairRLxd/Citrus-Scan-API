from enum import Enum


class ClassifierName(str, Enum):
    svm = "svm"
    bayes = "bayes"
    perceptron = "perceptron"

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

def simple_trick(learning_rate, weight_of_feature, feature, label, bias):
    predicted_label = weight_of_feature * feature + bias
    
    if label > predicted_label and label > 0:

        weight_of_feature += learning_rate
        bias += learning_rate

    elif label > predicted_label and label < 0:

        weight_of_feature -= learning_rate
        bias += learning_rate

    elif label < predicted_label and label > 0:

        weight_of_feature -= learning_rate
        bias -= learning_rate
    
    elif label < predicted_label and label < 0:

        weight_of_feature += learning_rate
        bias -= learning_rate


def absolute_trick(learning_rate, weight_of_feature, feature, label, bias):
    predicted_label = weight_of_feature * feature + bias
    
    # Case 1
    if predicted_label > label:
        weight_of_feature += learning_rate * feature
        bias += learning_rate

    # Case 2
    else:
        weight_of_feature -= learning_rate * feature
        bias -= learning_rate

    return weight_of_feature, bias


def square_trick(learning_rate, weight_of_feature, feature, label, bias):

    predicted_label = weight_of_feature * feature + bias
    weight_of_feature += learning_rate * feature * (label - predicted_label)
    bias += learning_rate * (label - predicted_label)

    return weight_of_feature, bias


def square_trick_multiple_feature(learning_rate, weights: list, features: list, label, bias):
    
    weights = np.array(weights)
    features = np.array(features)

    # p^ = b + w_1 * x_1 + w_2 * x_2 + w_3 * x_3 ........ + w_n * x_n 
    predicted_label = bias + np.dot(weights , features)
    bias += learning_rate * (label - predicted_label)

    for x in range(len(weights)):
        weights[x] += learning_rate * features[x] * (label - predicted_label)

    return weights, bias


def Linear_Regression(features, labels, learning_rate = None, epochs = None):
    
    weight = random.random()
    bias = random.random()

    if learning_rate is None:
        learning_rate = 0.01

    if epochs is None:
        epochs = 10000

    for i in range(epochs):
        # random number to choose a random point (x, y)

        n = random.randint(0, len(features) - 1)
        x = features[n]
        y = labels[n]

        # We will use the square trick
        weight, bias = square_trick(learning_rate, weight, x, y, bias)
        # weights, bias = simple_trick(learning_rate, weight, x, y, bias)
        # weight, bias = absolute_trick(learning_rate, weight, x, y, bias)

    return weight, bias


def Multiple_Feature_Linear_Regression(*features, labels, learning_rate = None, epochs = None):

    weights = [random.random() for x in range(len(features))]
    bias = random.random()

    if learning_rate is None:
        learning_rate = 0.01

    if epochs is None:
        epochs = 10000

    for i in range(epochs):
        # random number to choose a random point (x, y)
        
        n = random.randint(0, len(features) - 1)
        y = labels[n]

        x = list()
        for feature in features:
            x.append(feature[n])

        weight, bias = square_trick_multiple_feature(learning_rate, weight, x, y, bias)

    return weight, bias



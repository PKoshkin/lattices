from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from IPython.display import clear_output

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def read_file(filename):
    with open(filename) as handler:
        return [line.strip().split(", ") for line in handler]

def split_one_hot(values, splits):
    result = np.zeros((len(values), len(splits) - 1))
    for i, value in enumerate(values):
        for index, (min_bound, max_bound) in enumerate(zip(splits[:-1], splits[1:])):
            if (value >= min_bound) and (value < max_bound):
                result[i][index] = 1
                break
    return result

def split_one_hot_uniform(values, batches):
    minimum = np.min(values)
    maximum = np.max(values)
    splits = np.linspace(minimum, maximum, num=batches + 1)
    return split_one_hot(values, splits)

def equal_one_hot(values, positive_value):
    result = np.array([1, 0]).reshape([1, -1]).repeat(len(values), axis=0)
    positive_indices = (np.array(values) == positive_value)
    result[positive_indices, 0] = 0
    result[positive_indices, 1] = 1
    return result

def around_one_hot(values, middle_value):
    result = np.zeros((len(values), 3))
    middle_indices = (np.array(values) == middle_value)
    more_indices = (np.array(values) > middle_value)
    less_indices = (np.array(values) < middle_value)
    result[middle_indices, 1] = 1
    result[more_indices, 2] = 1
    result[less_indices, 0] = 1
    return result

def make_map(values):
    values_set = set(values)
    return {
        value: i
        for i, value in enumerate(values_set)
    }

def map_one_hot(values, map_to_label):
    max_label = len(map_to_label)
    result = np.zeros((len(values), max_label))
    for i, value in enumerate(values):
        result[i][map_to_label[value]] = 1
    return result

def map_one_hot_le(values, map_to_label):
    max_label = len(map_to_label)
    result = np.zeros((len(values), max_label))
    for i, value in enumerate(values):
        for j in range(i + 1):
            result[j][map_to_label[value]] = 1
    return result

def function_one_hot(values, function):
    labels = list(map(function, values))
    max_label = len(map_to_label)
    result = np.zeros((len(values), max_label))
    for i, label in enumerate(labels):
        result[i][label] = 1
    return result
    
def make_dataset(raw_data, average_positive_by_contry):
    columns = [
        [line[i] for line in raw_data]
        for i in range(len(raw_data[0]))
    ]

    # map one_hot
    one_hot_indices = [1, 5, 6, 7, 8, 9]
    maps = [
        make_map(columns[index])
        for index in one_hot_indices
    ]
    features = []
    for index, map_to_labels in zip(one_hot_indices, maps):
        features.append(map_one_hot(columns[index], map_to_labels))
    
    # le one_hot
    education_num_map = make_map(columns[4])
    features.append(map_one_hot(columns[4], education_num_map))
    
    # intervel one_hot
    batches_indices = [0]
    batches = [5]
    for index, batch in zip(batches_indices, batches):
        values = list(map(int, columns[index]))
        features.append(split_one_hot_uniform(values, batches=batch))

    # zero or not
    for index in [10, 11]:
        values = list(map(int, columns[index]))
        features.append(equal_one_hot(values, 0))
    function_indices = []
    
    # more, less or equal
    values = list(map(int, columns[12]))
    features.append(around_one_hot(values, 40))
    
    # contry

    def featrue_by_contry(country):
        if country == 'United-States':
            return [0, 1, 0]
        elif average_positive_by_contry[country] < average_positive_by_contry['United-States']:
            return [1, 0, 0]
        else:
            return [0, 0, 1]

    contry_features = np.array([
        featrue_by_contry(country)
        for country in columns[13]
    ])
    features.append(contry_features)
    
    targets = np.array([
        0 if target == '<=50K' else 1
        for target in columns[-1]
    ])
    
    return np.concatenate(features, axis=1), targets


def get_average_positive_target_rate(train):
    countries = set(line[13] for line in train)
    average_positive_target_rate = {}
    countries_positives = defaultdict(lambda: 0)
    countries_negatives = defaultdict(lambda: 0)
    for line in train:
        if line[-1] == '<=50K':
            countries_negatives[line[13]] += 1
        else:
            countries_positives[line[13]] += 1
    for country in countries:
        support = countries_positives[country] + countries_negatives[country]
        rate = float(countries_positives[country]) / support
        average_positive_target_rate[country] = rate
    return average_positive_target_rate


def get_catboost_acc(train_x, train_y, test_x, test_y):
    catboost_model = CatBoostClassifier(iterations=15)
    catboost_model.fit(train_x, train_y, verbose=0)
    catboost_predictions = catboost_model.predict(test_x)
    return accuracy_score(catboost_predictions, test_y)


def k_fold(X, Y, accuracy_calculator, folds=5):
    assert len(X) == len(Y)
    kf = KFold(n_splits=folds)
    evals = []
    for train_index, test_index in kf.split(X):
        train_x, test_x = X[train_index], X[test_index]
        train_y, test_y = Y[train_index], Y[test_index]
        evals.append(accuracy_calculator(train_x, train_y, test_x, test_y))
    return np.array(evals)


def unique_dataset(X, Y):
    unique_X_Y = set(map(tuple, np.concatenate([X, Y.reshape([-1, 1])], axis=1)))
    saved = set()
    bad_x = set()
    for x in map(tuple, unique_X_Y):
        x = x[:-1]
        if x in saved:
            bad_x.add(x)
        else:
            saved.add(x)
    unique_X_Y = np.array(list(unique_X_Y))
    X = unique_X_Y[:, :-1]
    Y = unique_X_Y[:, -1]
    indices = np.array([tuple(x) not in bad_x for x in X])
    return X[indices], Y[indices]


def to_set(features):
    return set(i for i, feature in enumerate(features) if feature == 1)


def lattices_stats(train_x, train_y, test_x):
    positives = [
        to_set(x)
        for x, y in zip(train_x, train_y)
        if y == 1
    ]
    negatives = [
        to_set(x)
        for x, y in zip(train_x, train_y)
        if y == 0
    ]
    test_x = [to_set(x) for x in test_x]
    
    def get_stats(positive_intersections, positives, negative_intersection, negatives, x):
        pure_positive_num = 0
        lens = []
        for intersection in positive_intersections:
            lens.append(len(intersection))
            for negative in negatives:
                if intersection <= negative:
                    break
            else:
                pure_positive_num += 1
        supports = []
        for intersection in positive_intersections + negative_intersection:
            support = 0
            for positive in positives:
                if intersection <= positive:
                    support += 1
            supports.append(support)
        return [pure_positive_num, np.mean(supports) / len(positives), np.mean(lens)]
    
    pos_stats = []
    neg_stats = []
    for i, x in enumerate(test_x):
        if i % 20 == 0:
            clear_output()
            print(i)

        positive_intersections = [positive & x for positive in positives]
        negative_intersections = [negative & x for negative in negatives]
        pos_stats.append(get_stats(
            positive_intersections,
            positives,
            negative_intersections,
            negatives,
            x
        ))
        neg_stats.append(get_stats(
            negative_intersections,
            negatives,
            positive_intersections,
            positives,
            x
        ))
    
    return np.array(pos_stats), np.array(neg_stats)
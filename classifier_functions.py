#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:49:45 2020

@author: jack
"""

# %%% Packages
import pandas as pd
import os
import numpy as np

from sklearn                         import tree
from sklearn                         import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model            import LinearRegression
from sklearn.neighbors               import KNeighborsClassifier
from sklearn.naive_bayes             import GaussianNB
from sklearn.model_selection         import RepeatedKFold

# Data splitting
def data_splitting(data):
    # Set random number seed
    np.random.seed(1)
    data['ML_group']   = np.random.randint(100, size = data.shape[0])
    data               = data.sort_values(by = 'ML_group')
    inx_train          = data.ML_group  <  80                                 # 80% of the data for training
    inx_valid          = (data.ML_group >= 80) & (data.ML_group<90)           # 10% for validating
    inx_test           = (data.ML_group >= 90)
    
    return data, inx_train, inx_valid, inx_test


# Putting structure in the review text
def structure_in_text(data, ngram_range, max_df, min_df):
    corpus      = data.review_text_actual.to_list()
    vectorizer  = CountVectorizer(lowercase  = True,
                             ngram_range = ngram_range,
                             max_df      = max_df,
                             min_df      = min_df);
    X = vectorizer.fit_transform(corpus)
    return X
    

# Performing the TVT split (Training, Validating, Testing)
def TVT_split(y_variable, x_variable, inx_train, inx_valid, inx_test):
    Y_train   = y_variable[inx_train].to_list()
    Y_valid   = y_variable[inx_valid].to_list()
    Y_test    = y_variable[inx_test].to_list()    

    X_train   = x_variable[np.where(inx_train)[0],:]
    X_valid   = x_variable[np.where(inx_valid)[0],:]
    X_test   = x_variable[np.where(inx_test) [0],:]
    
    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


def confusion_matrix(data, inx, Y_train, y_variable_name, pred_position):
    conf_matrix = np.zeros([max(Y_train), max(Y_train)])
    for i in range(max(Y_train)):
        for j in range(max(Y_train)):
            conf_matrix[i,j] = np.sum((data[inx][y_variable_name]==(i+1))*(data[inx][pred_position]==(j+1)))
    
    # Calculate how precise the matrix is, by summing the diagonal and diviving by the total
    right = conf_matrix.diagonal(offset = -1).sum() + conf_matrix.diagonal(offset = 0).sum() + conf_matrix.diagonal(offset = +1).sum()
    total = conf_matrix.sum()
    percent_accuracy = right/total
    
    return conf_matrix, percent_accuracy


def linear_reg(data, y_variable_name, Y_train, X_train, X_valid, X_test):
    results_list = []
    
    # Classify the data using the training data
    clf = LinearRegression().fit(X_train, Y_train)

    # Classifier predicts the results and stores them in a list    
    results_list.append(
        np.concatenate([
            clf.predict(X_train),
            clf.predict(X_valid),
            clf.predict(X_test)]
            ).round().astype(int))
    
    # Puts the results into a pandas DataFrame
    data_linear_reg                  = pd.DataFrame(results_list).transpose()
    data_linear_reg['inx_train']     = inx_train.to_list()
    data_linear_reg['inx_valid']     = inx_valid.to_list()
    data_linear_reg['inx_test']      = inx_test.to_list()
    data_linear_reg[y_variable_name] = data[y_variable_name].copy()

    # Replace predicted numbers out of index
    data_linear_reg.loc[data_linear_reg[0] > max(Y_train), 0] = max(Y_train)  
    data_linear_reg.loc[data_linear_reg[0] < min(Y_train), 0] = min(Y_train)
    
    # Confusion Matrix with the test data
    conf_matrix, percent_accuracy_test_linear_reg = confusion_matrix(data_linear_reg, inx_test, Y_train, y_variable_name, 0)
    
    return data_linear_reg, conf_matrix, percent_accuracy_test_linear_reg


# KNN Classifier
def KNN(data, Y_train, X_train, X_valid, X_test, y_variable_name, inx_train, inx_valid, inx_test):
    # Initialize variables
    k            = 1
    max_k_nn     = 100
    results_list = []
    
    # Loop through different number of neighbors
    for k in range(1, max_k_nn + 1):
    
        # Classify data only using the training data
        clf = KNeighborsClassifier(n_neighbors = k).fit(X_train, Y_train)
    
        # Classifier predicts the results and stores them in a list
        results_list.append(
            np.concatenate([
                clf.predict(X_train),
                clf.predict(X_valid),
                clf.predict(X_test)
                ]))

    # Puts the results into a pandas DataFrame
    data_knn                  = pd.DataFrame(results_list).transpose()
    data_knn['inx_train']     = inx_train.to_list()
    data_knn['inx_valid']     = inx_valid.to_list()
    data_knn['inx_test']      = inx_test.to_list()
    data_knn[y_variable_name] = data[y_variable_name].copy()

    # Initialize Variables for confusion matrices
    best_matrix_accuracy = 0
    conf_matrix_list = []

    # Confusion matrix for each k
    for k_neighbors in range(max_k_nn - 1):  
        
        # Confusion Matrix with validating data
        conf_matrix, percent_accuracy_valid_knn = confusion_matrix(data_knn, inx_valid, Y_train, y_variable_name, k_neighbors)
        
        conf_matrix_list.append(conf_matrix)

        # Stores the accuracy level and k of the most precise matrix
        if percent_accuracy_valid_knn > best_matrix_accuracy:
            best_matrix_accuracy = percent_accuracy_valid_knn.copy()
            best_k = k_neighbors + 1
    
    # Confusion Matrix with test data
    conf_matrix_test, percent_accuracy_test_knn = confusion_matrix(data_knn, inx_test, Y_train, y_variable_name, best_k)
    
    return data_knn, conf_matrix_test, percent_accuracy_test_knn, best_k


# Naive Bayes Classifier
def naive_bayes(data, Y_train, X_train, X_valid, X_test, y_variable_name, inx_train, inx_valid, inx_test):

    # Initialize the list where results will be stored
    results_list = []

    # Classify data using the training data
    clf = GaussianNB().fit(X_train.toarray(), Y_train)

    # Classifier predicts the results and stores them in a list
    results_list.append(
        np.concatenate([
                clf.predict(X_train.toarray()),
                clf.predict(X_valid.toarray()),
                clf.predict(X_test.toarray())
                ]))

    # Puts the results into a pandas DataFrame
    data_NB                  = pd.DataFrame(results_list).transpose()
    data_NB['inx_train']     = inx_train.to_list()
    data_NB['inx_valid']     = inx_valid.to_list()
    data_NB['inx_test']      = inx_test.to_list()
    data_NB[y_variable_name] = data[y_variable_name].copy()

    # Confusion matrix with test data
    conf_matrix_test, percent_accuracy_test_NB = confusion_matrix(data_NB, inx_test, Y_train, y_variable_name, 0)
    
    return data_NB, conf_matrix_test, percent_accuracy_test_NB


# Lasso Regression
def lasso_reg(data, Y_train, X_train, X_valid, X_test, y_variable_name, inx_train, inx_valid, inx_test):
    # Initialize the list where results will be stored
    results_list = []

    # Classify data using the training data
    clf = linear_model.Lasso(alpha = 0.1).fit(X_train, Y_train)

    # Classifier predicts the results and stores them in a list
    results_list.append(
        np.concatenate([
                clf.predict(X_train.toarray()),
                clf.predict(X_valid.toarray()),
                clf.predict(X_test.toarray())
                ]).round().astype(int))

    # Puts the results into a pandas DataFrame
    data_lasso_reg              = pd.DataFrame(results_list).transpose()
    data_lasso_reg['inx_train'] = inx_train.to_list()
    data_lasso_reg['inx_valid'] = inx_valid.to_list()
    data_lasso_reg['inx_test']  = inx_test.to_list()
    data_lasso_reg[y_variable_name] = data[y_variable_name].copy()

    # Confusion matrix with test data
    conf_matrix_test, percent_accuracy_test_lasso_reg = confusion_matrix(data_lasso_reg, inx_test, Y_train, y_variable_name, 0)
    
    return data_lasso_reg, conf_matrix_test, percent_accuracy_test_lasso_reg


# Lasso Cross Validation
def lasso_CV(data, Y_train, X_train, X_valid, X_test, y_variable_name, inx_train, inx_valid, inx_test):

    # Initialize the list where results will be stored
    results_list = []
    
    cv = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state = 1)

    # Classify data using the training data
    clf = linear_model.LassoCV(alphas = np.arange(0, 1, 0.01), cv = cv, n_jobs = -1).fit(X_train, Y_train)

    # Classifier predicts the results and stores them in a list
    results_list.append(
        np.concatenate([
                clf.predict(X_train.toarray()),
                clf.predict(X_valid.toarray()),
                clf.predict(X_test.toarray())
                ]).round().astype(int))

    # Puts the results into a pandas DataFrame
    data_lasso_CV              = pd.DataFrame(results_list).transpose()
    data_lasso_CV['inx_train'] = inx_train.to_list()
    data_lasso_CV['inx_valid'] = inx_valid.to_list()
    data_lasso_CV['inx_test']  = inx_test.to_list()
    data_lasso_CV[y_variable_name] = data[y_variable_name].copy()

    # Confusion matrix with test data
    conf_matrix_test, percent_accuracy_test_lasso_CV = confusion_matrix(data_lasso_CV, inx_test, Y_train, y_variable_name, 0)
    
    return data_lasso_CV, conf_matrix_test, percent_accuracy_test_lasso_CV, clf.alpha_


# Trees Classifier
def tree_classifier(data, Y_train, X_train, X_valid, X_test, y_variable_name, inx_train, inx_valid, inx_test):
    
    # Initialize Variables for classifier
    random_state         = 96
    max_depth            = 10 
    results_list         = []

    # Initialize Variables for confusion matrices
    best_matrix_accuracy = 0
    depth_counter = 0
    conf_matrix_list = []

    # Goes through both criterion and finds the best along the way
    for criterion in ['entropy','gini']:
    
        # Goes through all the depths and finds the best along the way    
        for depth in range(2, max_depth + 2):
        
            # Classify data using the training data
            clf = tree.DecisionTreeClassifier(
                criterion    = criterion, 
                max_depth    = depth,
                random_state = random_state).fit(X_train.toarray(), Y_train)
        
            # Classifier predicts the results and stores them in a list
            results_list.append(
                    np.concatenate([
                        clf.predict(X_train.toarray()),
                        clf.predict(X_valid.toarray()),
                        clf.predict(X_test.toarray())
                        ]))
        
    # Puts the results into a pandas DataFrame
    data_tree              = pd.DataFrame(results_list).transpose()
    data_tree['inx_train'] = inx_train.to_list()
    data_tree['inx_valid'] = inx_valid.to_list()
    data_tree['inx_test']  = inx_test.to_list()
    data_tree[y_variable_name] = data[y_variable_name].copy()
    
    # Confusion matrix with validating data
    conf_matrix, percent_accuracy_valid_tree = confusion_matrix(data_tree, inx_valid, Y_train, y_variable_name, depth_counter)
    conf_matrix_list.append(conf_matrix)
    
    # If statement to find the best hyperparameters, criterion and depth
    if percent_accuracy_valid_tree > best_matrix_accuracy:
        best_matrix_accuracy = percent_accuracy_valid_tree.copy()
        best_depth = depth_counter

        # If statement to know which criterion is best, and keeps track of the indexed position
        if criterion == 'entropy':
            best_criterion = 'entropy'
        elif criterion == 'gini':
            best_criterion = 'gini'
        
    # Keeps a count of what depth index position we are at
    depth_counter += 1
    
    # Confusion matrix with test data
    conf_matrix_test, percent_accuracy_test_tree = confusion_matrix(data_tree, inx_test, Y_train, y_variable_name, best_depth)

    return data_tree, conf_matrix_test, percent_accuracy_test_tree, best_depth, best_criterion


def precision_test(linear_reg, knn, NB, lasso_reg, lasso_CV, tree):
    most_precise = {
        'Linear Reg' : linear_reg,
        'KNN'        : knn,
        'Naive Bayes': NB,
        'Lasso Reg'  : lasso_reg,
        'Lasso CV'   : lasso_CV,
        'Tree'       : tree
        }
    
    maximum = max(most_precise, key = most_precise.get)
    percent_accuracy = most_precise[maximum]
    
    return maximum + ' ' + str(round(percent_accuracy*100,2)) + '%'

def sentiment(n):
    if n >= 4:
        return 3 # 3 for positive
    elif n == 3:
        return 2 # 2 for neutral
    else:
        return 1 # 1 for negative
    
    
# Functions for testing data with seperate training data
    
def confusion_matrix_2(data, Y_train, y_variable_name, pred_position):
    conf_matrix = np.zeros([max(Y_train), max(Y_train)])
    for i in range(max(Y_train)):
        for j in range(max(Y_train)):
            conf_matrix[i,j] = np.sum((data[y_variable_name]==(i+1))*(data[pred_position]==(j+1)))
    
    # Calculate how precise the matrix is, by summing the diagonal and diviving by the total
    right = conf_matrix.diagonal(offset = 0).sum()
    total = conf_matrix.sum()
    percent_accuracy = right/total
    
    return conf_matrix, percent_accuracy


def linear_reg_2(data, y_variable_name, Y_train, X_train, X):
    results_list = []
    
    # Classify the data using the training data
    clf = LinearRegression().fit(X_train, Y_train)

    # Classifier predicts the results and stores them in a list    
    results_list.append(
        np.concatenate([
            clf.predict(X)]
            ).round().astype(int))
    
    # Puts the results into a pandas DataFrame
    data_linear_reg                  = pd.DataFrame(results_list).transpose()
    #data_linear_reg['inx_train']     = inx_train.to_list()
    #data_linear_reg['inx_valid']     = inx_valid.to_list()
    #data_linear_reg['inx_test']      = inx_test.to_list()
    data_linear_reg[y_variable_name] = data[y_variable_name].copy()

    # Replace predicted numbers out of index
    data_linear_reg.loc[data_linear_reg[0] > max(Y_train), 0] = max(Y_train)  
    data_linear_reg.loc[data_linear_reg[0] < min(Y_train), 0] = min(Y_train)
    
    # Confusion Matrix with the test data
    conf_matrix, percent_accuracy_test_linear_reg = confusion_matrix_2(data_linear_reg, Y_train, y_variable_name, 0)
    
    return data_linear_reg, conf_matrix, percent_accuracy_test_linear_reg


# KNN Classifier
def KNN_2(data, data_2, Y_train, X_train, X_valid, X_test, y_variable_name, inx_train, inx_valid, inx_test, X):
    # Initialize variables
    k            = 1
    max_k_nn     = 100
    results_list = []
    results_list_2 = []
    
    # Loop through different number of neighbors
    for k in range(1, max_k_nn + 1):
    
        # Classify data only using the training data
        clf = KNeighborsClassifier(n_neighbors = k).fit(X_train, Y_train)
    
        # Classifier predicts the results and stores them in a list
        results_list.append(
            np.concatenate([
                clf.predict(X_train),
                clf.predict(X_valid),
                clf.predict(X_test)
                ]))
        
        results_list_2.append(
            np.concatenate([
                clf.predict(X)
                ]))

    # Puts the results into a pandas DataFrame
    data_knn                    = pd.DataFrame(results_list).transpose()
    data_knn['inx_train']       = inx_train.to_list()
    data_knn['inx_valid']       = inx_valid.to_list()
    data_knn['inx_test']        = inx_test.to_list()
    data_knn[y_variable_name]   = data[y_variable_name].copy()
    data_knn_2                  = pd.DataFrame(results_list_2).transpose()
    data_knn_2[y_variable_name] = data_2[y_variable_name].copy()

    # Initialize Variables for confusion matrices
    best_matrix_accuracy = 0
    conf_matrix_list = []

    # Confusion matrix for each k
    for k_neighbors in range(max_k_nn - 1):  
        
        # Confusion Matrix with validating data
        conf_matrix, percent_accuracy_valid_knn = confusion_matrix(data_knn, inx_valid, Y_train, y_variable_name, k_neighbors)
        
        conf_matrix_list.append(conf_matrix)

        # Stores the accuracy level and k of the most precise matrix
        if percent_accuracy_valid_knn > best_matrix_accuracy:
            best_matrix_accuracy = percent_accuracy_valid_knn.copy()
            best_k = k_neighbors + 1
    
    # Confusion Matrix with test data
    conf_matrix_test, percent_accuracy_test_knn = confusion_matrix_2(data_knn_2, Y_train, y_variable_name, best_k)
    
    return data_knn_2, conf_matrix_test, percent_accuracy_test_knn, best_k


# Naive Bayes Classifier
def naive_bayes_2(data, Y_train, X_train, X, y_variable_name):

    # Initialize the list where results will be stored
    results_list = []

    # Classify data using the training data
    clf = GaussianNB().fit(X_train.toarray(), Y_train)

    # Classifier predicts the results and stores them in a list
    results_list.append(
        np.concatenate([
                clf.predict(X.toarray())
                ]))

    # Puts the results into a pandas DataFrame
    data_NB                  = pd.DataFrame(results_list).transpose()
    #data_NB['inx_train']     = inx_train.to_list()
    #data_NB['inx_valid']     = inx_valid.to_list()
    #data_NB['inx_test']      = inx_test.to_list()
    data_NB[y_variable_name] = data[y_variable_name].copy()

    # Confusion matrix with test data
    conf_matrix_test, percent_accuracy_test_NB = confusion_matrix_2(data_NB, Y_train, y_variable_name, 0)
    
    return data_NB, conf_matrix_test, percent_accuracy_test_NB


# Lasso Regression
def lasso_reg_2(data, Y_train, X_train, X, y_variable_name):
    # Initialize the list where results will be stored
    results_list = []

    # Classify data using the training data
    clf = linear_model.Lasso(alpha = 0.1).fit(X_train, Y_train)

    # Classifier predicts the results and stores them in a list
    results_list.append(
        np.concatenate([
                clf.predict(X.toarray())
                ]).round().astype(int))

    # Puts the results into a pandas DataFrame
    data_lasso_reg              = pd.DataFrame(results_list).transpose()
    #data_lasso_reg['inx_train'] = inx_train.to_list()
    #data_lasso_reg['inx_valid'] = inx_valid.to_list()
    #data_lasso_reg['inx_test']  = inx_test.to_list()
    data_lasso_reg[y_variable_name] = data[y_variable_name].copy()

    # Confusion matrix with test data
    conf_matrix_test, percent_accuracy_test_lasso_reg = confusion_matrix_2(data_lasso_reg, Y_train, y_variable_name, 0)
    
    return data_lasso_reg, conf_matrix_test, percent_accuracy_test_lasso_reg


# Lasso Cross Validation
def lasso_CV_2(data, Y_train, X_train, X, y_variable_name):

    # Initialize the list where results will be stored
    results_list = []
    
    cv = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state = 1)

    # Classify data using the training data
    clf = linear_model.LassoCV(alphas = np.arange(0, 1, 0.01), cv = cv, n_jobs = -1).fit(X_train, Y_train)

    # Classifier predicts the results and stores them in a list
    results_list.append(
        np.concatenate([
                clf.predict(X.toarray())
                ]).round().astype(int))

    # Puts the results into a pandas DataFrame
    data_lasso_CV              = pd.DataFrame(results_list).transpose()
    #ata_lasso_CV['inx_train'] = inx_train.to_list()
    #data_lasso_CV['inx_valid'] = inx_valid.to_list()
    #data_lasso_CV['inx_test']  = inx_test.to_list()
    data_lasso_CV[y_variable_name] = data[y_variable_name].copy()

    # Confusion matrix with test data
    conf_matrix_test, percent_accuracy_test_lasso_CV = confusion_matrix_2(data_lasso_CV, Y_train, y_variable_name, 0)
    
    return data_lasso_CV, conf_matrix_test, percent_accuracy_test_lasso_CV, clf.alpha_


# Trees Classifier
def tree_classifier_2(data, data_2, Y_train, X_train, X_valid, X_test, X, y_variable_name, inx_train, inx_valid, inx_test):
    
    # Initialize Variables for classifier
    random_state         = 96
    max_depth            = 10 
    results_list         = []
    results_list_2       = []

    # Initialize Variables for confusion matrices
    best_matrix_accuracy = 0
    depth_counter = 0
    conf_matrix_list = []

    # Goes through both criterion and finds the best along the way
    for criterion in ['entropy','gini']:
    
        # Goes through all the depths and finds the best along the way    
        for depth in range(2, max_depth + 2):
        
            # Classify data using the training data
            clf = tree.DecisionTreeClassifier(
                criterion    = criterion, 
                max_depth    = depth,
                random_state = random_state).fit(X_train.toarray(), Y_train)
        
            # Classifier predicts the results and stores them in a list
            results_list.append(
                    np.concatenate([
                        clf.predict(X_train.toarray()),
                        clf.predict(X_valid.toarray()),
                        clf.predict(X_test.toarray())
                        ]))
            results_list_2.append(
                    np.concatenate([
                        clf.predict(X.toarray())
                        ]))
        
    # Puts the results into a pandas DataFrame
    data_tree              = pd.DataFrame(results_list).transpose()
    data_tree['inx_train'] = inx_train.to_list()
    data_tree['inx_valid'] = inx_valid.to_list()
    data_tree['inx_test']  = inx_test.to_list()
    data_tree[y_variable_name] = data[y_variable_name].copy()
    data_tree_2             = pd.DataFrame(results_list_2).transpose()
    data_tree_2[y_variable_name] = data_2[y_variable_name].copy()
    
    # Confusion matrix with validating data
    conf_matrix, percent_accuracy_valid_tree = confusion_matrix(data_tree, inx_valid, Y_train, y_variable_name, depth_counter)
    conf_matrix_list.append(conf_matrix)
    
    # If statement to find the best hyperparameters, criterion and depth
    if percent_accuracy_valid_tree > best_matrix_accuracy:
        best_matrix_accuracy = percent_accuracy_valid_tree.copy()
        best_depth = depth_counter

        # If statement to know which criterion is best, and keeps track of the indexed position
        if criterion == 'entropy':
            best_criterion = 'entropy'
        elif criterion == 'gini':
            best_criterion = 'gini'
        
    # Keeps a count of what depth index position we are at
    depth_counter += 1
    
    # Confusion matrix with test data
    conf_matrix_test, percent_accuracy_test_tree = confusion_matrix_2(data_tree_2, Y_train, y_variable_name, best_depth)

    return data_tree, conf_matrix_test, percent_accuracy_test_tree, best_depth, best_criterion
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 20:37:29 2020

@author: jack
"""

# Packages needed
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import classifier_functions as cf


# Loading the data
os.chdir('/Users/jack/Documents/BUS 256 Marketing Analytics/CARAWAY/')
caraway_tweets = pd.read_csv('caraway_tweets.csv').reset_index()
calphalon_tweets = pd.read_csv('calphalon_tweets.csv').reset_index()
data_caraway = pd.read_csv('caraway_reviews_dec7.csv').reset_index()

# Applying sentiment to caraway reviews, based off # of stars
data_caraway['sentiment'] = data_caraway.n_stars.apply(cf.sentiment)

# Putting strutcure in the text with CountVectorizer
corpus                   = data_caraway['review_text_actual'].to_list()
corpus_caraway_twitter   = caraway_tweets['2'].to_list()
corpus_calphalon_twitter = calphalon_tweets['2'].to_list()

vectorizer      = CountVectorizer(lowercase  = True,
                                 ngram_range = (1,1),
                                 max_df      = 0.85,
                                 min_df      = 0.01);
# X variables
X                  = vectorizer.fit_transform(corpus)
X_caraway_tweets   = vectorizer.transform(corpus_caraway_twitter)
X_calphalon_tweets = vectorizer.transform(corpus_calphalon_twitter)


# Data splitting
data_caraway, inx_train, inx_valid, inx_test = cf.data_splitting(data_caraway)

# TVT split
X_train_caraway, X_valid_caraway, X_test_caraway, Y_train_caraway, Y_valid_caraway, Y_test_caraway = cf.TVT_split(data_caraway.sentiment, X, inx_train, inx_valid, inx_test)


# Caraway Tweets
# Linear Regression
calphalonTW_linear_reg, conf_matrix_test_linear_reg_calphalonTW, percent_accuracy_test_linear_reg_calphalonTW = cf.linear_reg_2(calphalon_tweets, 'sentiment', Y_train_caraway, X_train_caraway, X_calphalon_tweets)

# KNN
calphalonTW_knn, conf_matrix_test_knn_calphalonTW, percent_accuracy_test_knn_calphalonTW, best_k = cf.KNN_2(data_caraway, calphalon_tweets, Y_train_caraway, X_train_caraway, X_valid_caraway, X_test_caraway, 'sentiment', inx_train, inx_valid, inx_test, X_calphalon_tweets)

# Naive Bayes
calphalonTW_NB, conf_matrix_test_NB_calphalonTW, percent_accuracy_test_NB_calphalonTW = cf.naive_bayes_2(calphalon_tweets, Y_train_caraway, X_train_caraway, X_calphalon_tweets, 'sentiment')

# Lasso Regression
calphalonTW_lasso_reg, conf_matrix_test_lasso_reg_calphalonTW, percent_accuracy_test_lasso_reg_calphalonTW = cf.lasso_reg_2(calphalon_tweets, Y_train_caraway, X_train_caraway, X_calphalon_tweets, 'sentiment')

# Lasso Cross-Validating
calphalonTW_lasso_CV, conf_matrix_test_lasso_CV_calphalonTW, percent_accuracy_test_lasso_CV_calphalonTW, chosen_alpha = cf.lasso_CV_2(calphalon_tweets, Y_train_caraway, X_train_caraway, X_calphalon_tweets, 'sentiment')

# Tree
calphalonTW_tree, conf_matrix_test_tree_calphalonTW, percent_accuracy_test_tree_calphalonTW, best_depth, best_criterion = cf.tree_classifier_2(data_caraway, calphalon_tweets, Y_train_caraway, X_train_caraway, X_valid_caraway, X_test_caraway, X_calphalon_tweets, 'sentiment', inx_train, inx_valid, inx_test)

# Precision Test
results = cf.precision_test(percent_accuracy_test_linear_reg_calphalonTW,
                      percent_accuracy_test_knn_calphalonTW,
                      percent_accuracy_test_NB_calphalonTW,
                      percent_accuracy_test_lasso_reg_calphalonTW,
                      percent_accuracy_test_lasso_CV_calphalonTW,
                      percent_accuracy_test_tree_calphalonTW)

print(results)


# Calphalon Tweets
# Linear Regression
carawayTW_linear_reg, conf_matrix_test_linear_reg_carawayTW, percent_accuracy_test_linear_reg_carawayTW = cf.linear_reg_2(caraway_tweets, 'sentiment', Y_train_caraway, X_train_caraway, X_caraway_tweets)

# KNN
carawayTW_knn, conf_matrix_test_knn_carawayTW, percent_accuracy_test_knn_carawayTW, best_k = cf.KNN_2(data_caraway, caraway_tweets, Y_train_caraway, X_train_caraway, X_valid_caraway, X_test_caraway, 'sentiment', inx_train, inx_valid, inx_test, X_caraway_tweets)

# Naive Bayes
carawayTW_NB, conf_matrix_test_NB_carawayTW, percent_accuracy_test_NB_carawayTW = cf.naive_bayes_2(caraway_tweets, Y_train_caraway, X_train_caraway, X_caraway_tweets, 'sentiment')

# Lasso Regression
carawayTW_lasso_reg, conf_matrix_test_lasso_reg_carawayTW, percent_accuracy_test_lasso_reg_carawayTW = cf.lasso_reg_2(caraway_tweets, Y_train_caraway, X_train_caraway, X_caraway_tweets, 'sentiment')

# Lasso Cross-Validating
carawayTW_lasso_CV, conf_matrix_test_lasso_CV_carawayTW, percent_accuracy_test_lasso_CV_carawayTW, chosen_alpha = cf.lasso_CV_2(caraway_tweets, Y_train_caraway, X_train_caraway, X_caraway_tweets, 'sentiment')

# Tree
carawayTW_tree, conf_matrix_test_tree_carawayTW, percent_accuracy_test_tree_carawayTW, best_depth, best_criterion = cf.tree_classifier_2(data_caraway, caraway_tweets, Y_train_caraway, X_train_caraway, X_valid_caraway, X_test_caraway, X_caraway_tweets, 'sentiment', inx_train, inx_valid, inx_test)

# Precision Test
results = cf.precision_test(percent_accuracy_test_linear_reg_carawayTW,
                      percent_accuracy_test_knn_carawayTW,
                      percent_accuracy_test_NB_carawayTW,
                      percent_accuracy_test_lasso_reg_carawayTW,
                      percent_accuracy_test_lasso_CV_carawayTW,
                      percent_accuracy_test_tree_carawayTW)

print(results)

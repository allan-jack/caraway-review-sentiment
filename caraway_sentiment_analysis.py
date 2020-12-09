import pandas as pd
import classifier_functions as cf

data_caraway = pd.read_csv('caraway_reviews_dec7.csv').reset_index()

data_caraway['sentiment'] = data_caraway.n_stars.apply(cf.sentiment)

# Data splitting
data_caraway, inx_train, inx_valid, inx_test = cf.data_splitting(data_caraway)

# structure into text
X = cf.structure_in_text(data_caraway, 'review_text_actual', (1,1), 0.85, 0.01)

# TVT split
X_train_caraway, X_valid_caraway, X_test_caraway, Y_train_caraway, Y_valid_caraway, Y_test_caraway = cf.TVT_split(data_caraway.sentiment, X, inx_train, inx_valid, inx_test)

# Linear Regression
data_caraway_linear_reg, conf_matrix_test_linear_reg_caraway, percent_accuracy_test_linear_reg_caraway = cf.linear_reg(data_caraway, 'sentiment', Y_train_caraway, X_train_caraway, X_valid_caraway, X_test_caraway)

# KNN
data_caraway_knn, conf_matrix_test_knn_caraway, percent_accuracy_test_knn_caraway, best_k = cf.KNN(data_caraway, Y_train_caraway, X_train_caraway, X_valid_caraway, X_test_caraway, 'sentiment', inx_train, inx_valid, inx_test)

# Naive Bayes
data_caraway_NB, conf_matrix_test_NB_caraway, percent_accuracy_test_NB_caraway = cf.naive_bayes(data_caraway, Y_train_caraway, X_train_caraway, X_valid_caraway, X_test_caraway, 'sentiment', inx_train, inx_valid, inx_test)

# Lasso Regression
data_caraway_lasso_reg, conf_matrix_test_lasso_reg_caraway, percent_accuracy_test_lasso_reg_caraway = cf.lasso_reg(data_caraway, Y_train_caraway, X_train_caraway, X_valid_caraway, X_test_caraway, 'sentiment', inx_train, inx_valid, inx_test)

# Lasso Cross-Validating
data_caraway_lasso_CV, conf_matrix_test_lasso_CV_caraway, percent_accuracy_test_lasso_CV_caraway, chosen_alpha = cf.lasso_CV(data_caraway, Y_train_caraway, X_train_caraway, X_valid_caraway, X_test_caraway, 'sentiment', inx_train, inx_valid, inx_test)

# Tree
data_caraway_tree, conf_matrix_test_tree_caraway, percent_accuracy_test_tree_caraway, best_depth, best_criterion = cf.tree_classifier(data_caraway, Y_train_caraway, X_train_caraway, X_valid_caraway, X_test_caraway, 'sentiment', inx_train, inx_valid, inx_test)

# Precision Test
results = cf.precision_test(percent_accuracy_test_linear_reg_caraway,
                      percent_accuracy_test_knn_caraway,
                      percent_accuracy_test_NB_caraway,
                      percent_accuracy_test_lasso_reg_caraway,
                      percent_accuracy_test_lasso_CV_caraway,
                      percent_accuracy_test_tree_caraway)

print(results)
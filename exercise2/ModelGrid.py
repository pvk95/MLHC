import pickle
import numpy as np
from helper import EstimatorSelectionHelper
from sklearn.metrics import f1_score, make_scorer, fbeta_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

dir = 'final_data/noPrep/uni/'

x_train_dir = dir + 'X_train.pkl'
# x_train_dir = 'C:/Users/juaaa/PycharmProjects/real_active_learning/data/X_train.pkl'
with open(x_train_dir, 'rb') as input_file:
    x_train = pickle.load(input_file)
# x_train = x_train[:846]

y_train_dir = dir + 'y_train.pkl'
# y_train_dir = 'C:/Users/juaaa/PycharmProjects/real_active_learning/data/y_train.pkl'
with open(y_train_dir, 'rb') as input_file:
    y_train = pickle.load(input_file)
y_train = y_train.ravel().astype(np.int64)
# y_train = y_train[:846]

x_test_dir = dir + 'X_test.pkl'
# x_test_dir = 'C:/Users/juaaa/PycharmProjects/real_active_learning/data/X_test.pkl'
with open(x_test_dir, 'rb') as input_file:
    x_test = pickle.load(input_file)

y_test_dir = dir + 'y_test.pkl'
# y_test_dir = 'C:/Users/juaaa/PycharmProjects/real_active_learning/data/y_test.pkl'
with open(y_test_dir, 'rb') as input_file:
    y_test = pickle.load(input_file)
y_test = y_test.ravel().astype(np.int64)


models = {
    'AdaBoostClassifier': AdaBoostClassifier(base_estimator=LogisticRegression(solver='liblinear',
                                                                               class_weight='balanced')),
    'SVC': SVC(class_weight='balanced', gamma='auto'),
    'LogisticRegression': LogisticRegression(solver='liblinear', class_weight='balanced'),
    'GaussianNB': GaussianNB(),
    'BernoulliNB': BernoulliNB(),
    'RandomForest': RandomForestClassifier(class_weight='balanced', max_depth=5)
}

params = {
    'AdaBoostClassifier':  {'n_estimators': [8, 16, 32, 64, 128, 256]},
    'SVC': [
        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
        {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100]},
    ],
    'LogisticRegression': {'C': [0.1, 1, 10, 50, 100]},
    'GaussianNB': {},
    'BernoulliNB': {},
    'RandomForest': {'n_estimators': [16, 32, 100]},
}
if __name__ == "__main__":
    helper = EstimatorSelectionHelper(models, params)
    helper.fit(x_train, y_train, x_test, y_test, scoring=make_scorer(f1_score), n_jobs=-1)
    summary = helper.score_summary(sort_by='mean_score')
    # summary.to_pickle('summary')
    sortedKeysAndValues = sorted(helper.test_scores.items(), key=lambda kv: -kv[1])
    for k, v in sortedKeysAndValues:
        print(k + ': ' + str(v))

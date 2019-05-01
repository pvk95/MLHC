import pandas as pd
import os
import sys
import models
import types
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

gpu = 0
lstm_out = 100
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

df_1 = pd.read_csv("exercise_data/heartbeat/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("exercise_data/heartbeat/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(186))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(186))].values)[..., np.newaxis]

metrics_df = pd.DataFrame(data=[],columns=['Name','f1_score','AUROC','AUPRC','ACC'])

models_ = [
    models.LSTM_Model(),
    RandomForestClassifier(n_jobs=-1),
    models.Residual_CNN(),
    models.CNN_Model(),
]

params = [
    # LSTM
    {
        'verbose': [1],
        'hidden': [16, 32, 64],
        'dense': [16, 32, 64]
    },
    # RandomForestClassifier
    {
        'n_estimators' : [10, 100, 200],
        'n_jobs':  [-1]
    },
    # Residual_CNN
    {
        'deepness': range(1,6),
        'verbose': [0]
    },
    # CNN_Model
    {   
        'conv1_size': [16, 32],
        'conv2_size': [32, 64],
        'conv3_size': [128, 256],
        'dense_size': [16, 32, 64],
        'verbose': [0]
    },
    
]

for param, model in zip(params, models_):
    clf = RandomizedSearchCV(model, param, cv=2, n_jobs=1)
    if type(model) == RandomForestClassifier:
        clf.fit(np.squeeze(X), Y)
        model = clf.best_estimator_
        model.getScores = types.MethodType(models.CNN_Model.getScores, model)
        metrics_df = model.getScores(np.squeeze(X_test), Y_test, metrics_df)
    else:
        clf.fit(X, Y)
        model = clf.best_estimator_
        metrics_df = model.getScores(X_test, Y_test, metrics_df)
    print(metrics_df)

print(metrics_df)

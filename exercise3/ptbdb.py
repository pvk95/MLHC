import pandas as pd
import os
import sys
import models
import numpy as np
from sklearn.model_selection import train_test_split

gpu = 0
lstm_out = 100
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

df_1 = pd.read_csv("exercise_data/heartbeat/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("exercise_data/heartbeat/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

metrics_df = pd.DataFrame(data=[],columns=['Name','f1_score','AUROC','AUPRC','ACC'])

models_ = [
    models.Residual_CNN('residual'),
    models.CNN_Model('baseline'),
    models.LSTM_Model('LSTM', epochs=1)
]

for model in models_:
    model.train(X, Y, f'{model.name}_ptbdb.h5')
    metrics_df = model.getScores(X_test, Y_test, metrics_df)

print(metrics_df)
import pandas as pd
import os
import sys
import models
import types
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import sklearn.mixture
from sklearn.metrics import roc_curve,precision_recall_curve,auc,accuracy_score,f1_score, roc_auc_score
import matplotlib.pyplot as plt

def getScores(model_name,Y_test,pred_test,metrics_df, multilabel=False):
    if multilabel:
        pred_test = keras.utils.to_categorical(pred_test)

    # compute average roc_auc_score
    auroc = roc_auc_score(Y_test, pred_test)

    # skip precsion recall in multilabel-case
    if multilabel:
        auprc = 0
    else:
        precision, recall, _ = precision_recall_curve(Y_test, pred_test)
        auprc = auc(recall, precision)

    pred_test = (pred_test > 0.5).astype(np.int8)

    # compute f1_score
    if multilabel:
        f1 = f1_score(Y_test, pred_test, average='micro')
    else:
        f1 = f1_score(Y_test, pred_test)

    # comput accuracy
    acc = accuracy_score(Y_test, pred_test)
    curr_metrics = {'Name':model_name,'f1_score': f1, "AUROC": auroc, "AUPRC": auprc, "ACC": acc}
    metrics_df = metrics_df.append(curr_metrics, ignore_index = True)
    return metrics_df

def visualize(df,title):
    plt.figure(figsize=(10,8))
    np.random.seed(0)
    n_sub_plots = 5
    for i in range(n_sub_plots):
        plt.subplot(n_sub_plots, 1, i + 1)
        plt.plot(df.iloc[np.random.choice(len(df[1])), :])
        plt.xticks([])
        plt.yticks([])

    plt.suptitle(title)
    #plt.show()


gpu = 7
lstm_out = 100
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

df_1 = pd.read_csv("exercise_data/heartbeat/mitbih_train.csv", header=None)
df_2 = pd.read_csv("exercise_data/heartbeat/mitbih_test.csv", header=None)

visualize(df_1, 'Training Set EEG')

df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

# 87556 samples
Y = np.array(df_train[187].values).astype(np.int8)
Y = keras.utils.to_categorical(Y)
np.shape(Y)
X = np.array(df_train[list(range(186))].values)[..., np.newaxis]

# 21890 samples
Y_test = np.array(df_test[187].values).astype(np.int8)
Y_test = keras.utils.to_categorical(Y_test)
np.shape(Y_test)
X_test = np.array(df_test[list(range(186))].values)[..., np.newaxis]

metrics_df = pd.DataFrame(data=[],columns=['Name','f1_score','AUROC','AUPRC','ACC'])

models_ = [
    sklearn.mixture.GaussianMixture(n_components=5),
    sklearn.mixture.BayesianGaussianMixture(n_components=5),
    models.LSTM_Model(outputs=5),
    RandomForestClassifier(n_jobs=-1),
    models.Residual_CNN(outputs=5),
    models.CNN_Model(outputs=5),
]

params = [
    # Gaussian mixture
    {

    },
    # Bayesian Mixture
    {

    },
    # LSTM
    {
        'hidden': [16, 32, 64],
        'dense': [16, 32, 64]
    },
    # RandomForestClassifier
    {
        'n_estimators': [10, 100, 200],
        'n_jobs': [-1]
    },
    # Residual_CNN
    {
        'deepness': range(1, 6),
    },
    # CNN_Model
    {
        'conv1_size': [16, 32],
        'conv2_size': [32, 64],
        'conv3_size': [128, 256],
        'dense_size': [16, 32, 64],
    },

]

model_preds = []
for param, model in zip(params, models_):
    clf = RandomizedSearchCV(model, param, cv=2,n_iter=5)
    if type(model) == RandomForestClassifier or \
        type(model) == sklearn.mixture.GaussianMixture or \
        type(model) == sklearn.mixture.BayesianGaussianMixture:
        clf.fit(np.squeeze(X), Y)
        model = clf.best_estimator_
        model.getScores = types.MethodType(models.CNN_Model.getScores, model)
        _,metrics_df = model.getScores(np.squeeze(X_test), Y_test, metrics_df, multilabel=True)
    else:
        clf.fit(X, Y)
        model = clf.best_estimator_
        pred,metrics_df = model.getScores(X_test, Y_test, metrics_df)
        model_preds.append(pred)
    print(metrics_df)

model_preds = np.array(model_preds)
model_preds = np.squeeze(model_preds)

#Avg ensemble:
avg_pred = np.mean(model_preds,axis=0)
metrics_df = getScores('Ensemble(Avg)',Y_test=Y_test,pred_test=avg_pred,metrics_df=metrics_df, multilabel=True)

## MF: not applicable for multi-class
# #Logistic regression
# from sklearn.linear_model import LogisticRegression
# lg = LogisticRegression(n_jobs=-1)
# X_lg = np.transpose(model_preds,[1,0])
# lg.fit(X_lg,Y_test)
# lg_pred = lg.predict_proba(X_lg)[:,0]
# metrics_df = getScores('Ensemble(LG)',Y_test=Y_test,pred_test=lg_pred,metrics_df=metrics_df)
#
print(metrics_df)

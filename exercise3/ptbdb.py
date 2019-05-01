import pandas as pd
import os
import models
import types
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import sklearn.mixture
from sklearn.metrics import roc_curve,precision_recall_curve,auc,accuracy_score,f1_score
import matplotlib.pyplot as plt

def getScores(model_name,Y_test,pred_test,metrics_df):
    fpr, tpr, _ = roc_curve(Y_test, pred_test)
    auroc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(Y_test, pred_test)
    auprc = auc(recall, precision)
    pred_test = (pred_test > 0.5).astype(np.int8)
    f1 = f1_score(Y_test, pred_test)
    acc = accuracy_score(Y_test, pred_test)
    curr_metrics = {'Name': model_name, 'f1_score': f1, "AUROC": auroc, "AUPRC": auprc, "ACC": acc}
    metrics_df = metrics_df.append(curr_metrics, ignore_index=True)
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

df_1 = pd.read_csv("exercise_data/heartbeat/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("exercise_data/heartbeat/ptbdb_abnormal.csv", header=None)

visualize(df_1,'Normal EEG')
visualize(df_2,'Abnormal EEG')

df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(186))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(186))].values)[..., np.newaxis]

#X_ft = standard.getfeatures(X)
#X_test_ft = standard.getfeatures(X_test)

metrics_df = pd.DataFrame(data=[],columns=['Name','f1_score','AUROC','AUPRC','ACC'])

models_ = [
    sklearn.mixture.GaussianMixture(n_components=2),
    sklearn.mixture.BayesianGaussianMixture(n_components=2),
    models.LSTM_Model(),
    RandomForestClassifier(n_jobs=-1),
    models.Residual_CNN(),
    models.CNN_Model(),
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
        'n_estimators' : [10, 100, 200],
        'n_jobs':  [-1]
    },
    # Residual_CNN
    {
        'deepness': range(1,6),
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
        _,metrics_df = model.getScores(np.squeeze(X_test), Y_test, metrics_df)
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
metrics_df = getScores('Ensemble(Avg)',Y_test=Y_test,pred_test=avg_pred,metrics_df=metrics_df)

#Logistic regression
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(n_jobs=-1)
X_lg = np.transpose(model_preds,[1,0])
lg.fit(X_lg,Y_test)
lg_pred = lg.predict_proba(X_lg)[:,0]
metrics_df = getScores('Ensemble(LG)',Y_test=Y_test,pred_test=lg_pred,metrics_df=metrics_df)

print(metrics_df)
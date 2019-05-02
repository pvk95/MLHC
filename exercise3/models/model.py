from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score, roc_auc_score
import numpy as np
import sklearn.mixture

class Model(object):
    def __init__(self, input_shape, outputs, epochs, summary, verbose):
        self.input_shape = input_shape
        self.outputs = outputs
        self.epochs = epochs
        self.summary = summary
        self.verbose = verbose
        self.model = None

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        d = vars(self)
        d = dict(d)
        d.pop('model')
        return d

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X, Y):
        checkpoint = ModelCheckpoint(type(self).__name__ + '.h5', monitor='val_acc', verbose=self.verbose, save_best_only=True, mode='max')
        early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=self.verbose)
        redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=self.verbose)
        callbacks_list = [checkpoint, early, redonplat]  # early

        #self.model.fit(X, Y, epochs=1, verbose=self.verbose, callbacks=callbacks_list, validation_split=0.1)
        self.model.fit(X, Y, epochs=self.epochs, verbose=self.verbose, callbacks=callbacks_list, validation_split=0.1)
        #self.model.load_weights(type(self).__name__ + '.h5')
        return self
    
    def score(self, X, Y):
        return self.model.evaluate(X,Y)[1]

    def getScores(self, X_test, Y_test, metrics_df, multilabel=False, eval_train=False):
        pred_test = np.squeeze(self.predict(X_test))
        pred = pred_test.copy()

        if multilabel:
           pred_test = to_categorical(pred_test)

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
        curr_metrics = {'Name':type(self).__name__,'f1_score': f1, "AUROC": auroc, "AUPRC": auprc, "ACC": acc}
        if not eval_train:
            metrics_df = metrics_df.append(curr_metrics, ignore_index = True)
        return pred, metrics_df

    def getScores_multi(self, X_test, Y_test, metrics_df):
        if type(self) == sklearn.mixture.GaussianMixture or \
            type(self) == sklearn.mixture.BayesianGaussianMixture:
            pred_test = self.predict_proba(X_test)
            pred_test_temp = np.argmax(pred_test, axis=-1)
        elif type(self) == sklearn.ensemble.RandomForestClassifier:
            pred_test = self.predict_proba(X_test)
            pred_test = np.array(pred_test)[:,:,1]
            pred_test_temp = np.argmax(pred_test, axis=0)
        else:
            pred_test = (self.predict(X_test))
            pred_test_temp = np.argmax(pred_test,axis=-1)

        pred = pred_test.copy()
        Y_test_temp = np.argmax(Y_test, axis=-1)
        acc = accuracy_score(Y_test_temp, pred_test_temp)
        curr_metrics = {'Name':type(self).__name__,"ACC": acc}
        metrics_df = metrics_df.append(curr_metrics,ignore_index = True)
        return pred,metrics_df

'''
Boilerplate for a new class

class Model_Name(Model):
    def __init__(self, name='Model', input_shape=(187, 1), outputs=1, epochs=1000, summary=False):
        super().__init__(name, input_shape, outputs, epochs, summary)
        self.model = self.getModel()
    
    def getModel(self):
        pass
'''



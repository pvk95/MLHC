from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score
import numpy as np

class Model(object):
    def __init__(self, name, input_shape, outputs, epochs, summary):
        self.name = name
        self.input_shape = input_shape
        self.outputs = outputs
        self.epochs = epochs
        self.summary = summary
        self.model = None

    def train(self, X, Y, file_path):
        checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
        redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
        callbacks_list = [checkpoint, early, redonplat]  # early

        self.model.fit(X, Y, epochs=self.epochs, verbose=1, callbacks=callbacks_list, validation_split=0.1)
        self.model.load_weights(file_path)
        return self

    def getScores(self, X_test, Y_test, metrics_df):
        pred_test = self.model.predict(X_test)
        fpr, tpr, _ = roc_curve(Y_test, pred_test)
        auroc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(Y_test, pred_test)
        auprc = auc(recall, precision)
        pred_test = (pred_test > 0.5).astype(np.int8)
        f1 = f1_score(Y_test, pred_test)
        acc = accuracy_score(Y_test, pred_test)
        curr_metrics = {'Name':self.name,'f1_score': f1, "AUROC": auroc, "AUPRC": auprc, "ACC": acc}
        metrics_df = metrics_df.append(curr_metrics,ignore_index = True)
        return metrics_df



'''
Boilerplate for a new class

class Model_Name(Model):
    def __init__(self, name='Model', input_shape=(187, 1), outputs=1, epochs=1000, summary=False):
        super().__init__(name, input_shape, outputs, epochs, summary)
        self.model = self.getModel()
    
    def getModel(self):
        pass
'''



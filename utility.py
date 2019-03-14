import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score

mapping = {'A': [1, 0, 0, 0], 
           'T': [0, 1, 0, 0],
           'C': [0, 0, 1, 0],
           'G': [0, 0, 0, 1]
}


def map_dna_into_vector(string):
    '''
        This function maps a DNA string (ATCG) into a one-hot encoded vector
    '''
    vector = []
    for c in string:
        vector.append(mapping[c])
    vector = np.hstack(vector)
    return vector

def get_scores(true_val, pred_val):    
    '''
        computes the Area under the ROC-Curve and the area under the Precision-Recall-Curve
    '''
    fpr, tpr, _ = roc_curve(true_val, pred_val)
    auroc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(true_val, pred_val)
    auprc = auc(recall, precision)
    
    f1 = f1_score(true_val, pred_val)
    return (auroc, auprc, f1)
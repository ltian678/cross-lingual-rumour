import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support
from collections import defaultdict
implemented_metrics = ['acc', 'prec', 'rec', 'f1']

class Evaluator:
    # Evaluator for all evaluation metrics
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.metric = args.metric

    def evaluate(self, preds, golds, proba=None, verbose=True):
        preds = np.array(preds)
        golds = np.array(golds)

        total_num = len(preds)

        pred = list(preds)
        true = list(golds)
        acc = accuracy_score(y_true=true, y_pred=pred)
        f1 = f1_score(y_true=true, y_pred=pred, average='macro')
        prec, rec, fscore, support = precision_recall_fscore_support(y_true=true, y_pred=pred, average='macro')
        conf_mat = confusion_matrix(y_true=true, y_pred=pred)
        clf_report = classification_report(y_true=true, y_pred=pred)


        res = {
            'acc': acc,
            'prec': prec,
            'rec': rec,
            'f1': f1,
            'total': total_num,
            'confusion_matrix': conf_mat,
            'classification_report': clf_report
        }

        return res

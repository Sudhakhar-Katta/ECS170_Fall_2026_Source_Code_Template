'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')

        true_y=self.data["true_y"]
        pred_y=self.data["pred_y"]

        accuracy=accuracy_score(true_y,pred_y)

        precision= precision_score(
            true_y, pred_y,
            average='macro',
            zero_division=0
        )

        recall= recall_score(
            true_y, pred_y,
            average='macro',
            zero_division=0
        )

        f1=f1_score(
            true_y,pred_y,
            average='macro',
            zero_division=0
        )

        self.metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

        return accuracy
        
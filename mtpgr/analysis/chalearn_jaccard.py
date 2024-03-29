from sklearn.metrics import jaccard_score
import numpy as np

class ChaLearnJaccard:
    """
    The metric defined in ChaLearn's paper: 
    ChaLearn Looking at People RGB-D Isolated and Continuous Datasets for Gesture Recognition
    """
    def __init__(self, num_classes) -> None:
        self.L = num_classes  # L: the total number of gesture classes in the task.

    def mean_jaccard_index(self, gt_pred_list):
        """
        gt_pred_list: List of (gt, pred)
            [
                (gt[0, 1, 2], pred[2, 1, 0]),
                (gt[1, 0, 1], pred[3, 0, 1]),
                ...
            ]
        """

        # Equation 4
        j_single_seq = [self._sequence_jaccard(gt, pred) for gt, pred in gt_pred_list]
        j_matrix = np.array([d["each"] for d in j_single_seq])  # Shape: (num_seq, num_class)
        Jsj: np.ndarray = np.array([d["sum"] for d in j_single_seq])  # Shape: (num_seq)
        Js_bar = Jsj.mean()  # Shape: (,)
        return Js_bar, Jsj, j_matrix

    def _sequence_jaccard(self, y_true, y_pred):
        # Equation 2
        j: np.ndarray = jaccard_score(y_true, y_pred, labels=range(self.L), average=None, zero_division=0)  # j: scores for each class. array([1. , 0. , 0.33...])

        # Equation 3
        # ls: the number of gesture classes appeared in the y_true sequence.
        ls = len(set(y_true))
        Js = j.sum() / ls

        return {"sum": Js, "each": j}

if __name__ == "__main__":

    gplist = []
    gplist.append(([0, 0, 0], [0, 1, 2]))
    gplist.append(([5, 5, 5], [5, 4, 4]))
    jaccard = ChaLearnJaccard(8).mean_jaccard_index(gplist)
    print("Testing ChaLearn Jaccard Index")
    print(jaccard)
    
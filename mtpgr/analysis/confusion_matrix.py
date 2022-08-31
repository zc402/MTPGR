
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix

def compute_cm(y_true, y_pred, save_folder:Path=None):
    cm = confusion_matrix(y_true, y_pred, )
    # np.set_printoptions(precision=2)
    # print('Confusion matrix, without normalization')
    # print(cm)
    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print('Normalized confusion matrix')
    # print(cm_normalized)

    if save_folder is not None:
        np.savetxt(save_folder / "cm.txt", cm, fmt='%-6d')
        np.savetxt(save_folder / "cm_norm.txt", cm_normalized, fmt='%.2f')
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    # plot_confusion_matrix(cm)
    plt.savefig(save_folder / "cm_norm.pdf")


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar
    tick_marks = np.arange(len(range(33)))
    plt.xticks(tick_marks, range(33), rotation=45)
    plt.yticks(tick_marks, range(33))
    plt.tight_layout
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



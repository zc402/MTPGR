
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def compute_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, )
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    # plot_confusion_matrix(cm)
    plt.show()


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



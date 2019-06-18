"""
Accuracy rates

Expects 2 objects: lists, numpy arrays or pandas
pred: contains prediction labels
labels: contains true labels

Usage

fp = false_pos(pred, labels)
tn = true_neg(pred, labels)
fn = false_neg(pred, labels)
tp = true_pos(pred, labels)


"""


def true_pos(pred, labels):
    return ((pred == labels) & (pred == 1)).sum()


def true_neg(pred, labels):
    return ((pred == labels) & (pred == 0)).sum()


def false_pos(pred, labels):
    return ((pred != labels) & (pred == 1)).sum()


def false_neg(pred, labels):
    return ((pred != labels) & (pred == 0)).sum()


def pr(pred):
    return pred.sum() / len(pred)


def tpr(pred, labels):
    """
    RECALL

    pred is positive and true given positive label
    P[pred == label | label == 1]
    =
    P[pred == 1 | label == 1]

    (1,1) / (x, 1)
    (1,1) / (1,1) + (0,1)
    tp / tp + fn
    """
    # true positive / labels==1, i.e. tp + fn
    tp = true_pos(pred, labels)
    fn = false_neg(pred, labels)
    return tp / (tp + fn)


def fpr(pred, labels):
    """
    pred is positive and false given negative label
    P[pred != label | label == 0]
    =
    P[pred == 1 | label == 0]

    (1,0) / (x, 0)
    (1,0) / (0,0) + (1,0)
    fp / tn + fp
    """
    fp = false_pos(pred, labels)
    tn = true_neg(pred, labels)
    return fp / (tn + fp)


def fnr(pred, labels):
    """
    pred is negative and false given positive label
    P[pred != label | label == 1]
    =
    P[pred == 0 | label == 1]

    (0,1) / (x, 1)
    (0,1) / (0,1) + (1,1)
    fn / fn + tp
    """
    fn = false_neg(pred, labels)
    tp = true_pos(pred, labels)
    return fn / (fn + tp)


def tnr(pred, labels):
    """
    pred is negative and true given negative label
    P[pred == label | label == 0]
    =
    P[pred == 0 | label == 0]

    (0,0) / (x, 0)
    (0,0) / (0,0) + (1,0)
    tn / tn + fp
    """
    tn = true_neg(pred, labels)
    fp = false_pos(pred, labels)
    return tn / (tn + fp)


def ppv(pred, labels):
    """
    PRECISION

    label is positive given a positive pred
    P[label == 1| pred == 1]

    (1,1) / (1, x)
    (1,1) / (1,0) + (1,1)
    tp / fp + tp
    """
    tp = true_pos(pred, labels)
    fp = false_pos(pred, labels)
    return tp / (tp + fp)


def npv(pred, labels):
    """
    label is negative given a negative pred
    P[label == 0| pred == 0]

    (0,0) / (0, x)
    (0,0) / (0,0) + (0,1)
    tn / tn + fn
    """
    tn = true_neg(pred, labels)
    fn = false_neg(pred, labels)
    return tn / (tn + fn)

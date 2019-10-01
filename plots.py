
import matplotlib.patches as mpatches
from dataset import importMeta
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def printPlots(classes, dataset_dir, epochs, train_loss, train_acc, test_loss, test_acc, predict, true):

    train_labels = importMeta(dataset_dir / 'train')
    train_classes = list(0. for i in range(10))
    for l in train_labels:
        train_classes[l] += 1

    test_labels = importMeta(dataset_dir / 'test')
    test_classes = list(0. for i in range(10))
    for l in test_labels:
        test_classes[l] += 1

    #datasetDistribution(train_classes, classes, 'Trainset')
    #datasetDistribution(test_classes, classes, 'Testset')
    datasetComparison(classes, train_classes, test_classes)

    #lossHistory('Train', train_loss)
    #accuracyHistory('Train', train_acc)

    #lossHistory('Test', test_loss)
    #accuracyHistory('Test', test_acc)

    #netAccuracy(classes, class_acc)
    lossComparison(train_loss, test_loss, epochs)
    accComparison(train_acc, test_acc, epochs)

    conf_matrix1(true, predict, classes, list(i for i in range(10)), figsize=(10, 10))
    conf_matrix2(true, predict, classes, list(i for i in range(10)))

def datasetDistribution(labels, classes, title):
    plt.bar(classes, labels)
    plt.xticks(rotation=30, ha='right')
    plt.title(f"{title} distribution")
    plt.savefig('dataset_distribution.png')
    plt.show()

def datasetComparison(classes, train, test):
    # plt.bar(classes, train)
    # plt.bar(classes, test)
    # plt.xticks(rotation=30, ha='right')
    #
    # plt.savefig('dataset_comparison')
    # plt.show()
    plt.title('Trainset and testset distribution', fontweight='bold')
    barWidth = 0.5

    r0 = np.arange(len(classes))
    r1 = [x - barWidth/2 for x in r0]
    r2 = [x + barWidth/2 for x in r0]

    plt.bar(r1, train, color='orange', width=barWidth, edgecolor='white', label='train')
    plt.bar(r2, test, color='royalblue', width=barWidth, edgecolor='white', label='test')

    plt.xlabel('Classes', fontweight='bold')
    plt.ylabel('N° of images', fontweight='bold')
    plt.xticks(r0, classes)
    plt.xticks(rotation=30, ha='right')
    plt.legend()
    plt.savefig('dataset_comparison')
    plt.show()


def lossHistory(name, losses):
    plt.plot(losses, linewidth=4)
    plt.title(name + ' loss history', fontweight='bold')
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.savefig(name + '_loss_history.png')
    plt.show()

def lossComparison(train_loss, test_loss, epochs):
    trainig = mpatches.Patch(color='royalblue', label='Training')
    testing = mpatches.Patch(color='orange', label='Testing')
    plt.plot(np.arange(epochs), train_loss, np.arange(epochs), test_loss, linewidth=4)
    plt.title('Loss comparison', fontweight='bold')
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.legend(handles=[trainig, testing])
    plt.savefig('loss_comparison.png')
    plt.show()

def accComparison(train_acc, test_acc, epochs):
    trainig = mpatches.Patch(color='royalblue', label='Training')
    testing = mpatches.Patch(color='orange', label='Testing')
    plt.plot(np.arange(epochs), train_acc, np.arange(epochs), test_acc, linewidth=4)
    plt.title('Accuracy comparison', fontweight='bold')
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.legend(handles=[trainig, testing])
    plt.savefig('acc_comparison.png')
    plt.show()

def accuracyHistory(name, accs):
    plt.plot(accs, linewidth=4)
    plt.title(name + ' accuracy history', fontweight='bold')
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.savefig(name + '_acc_history.png')
    plt.show()

def netAccuracy(classes, accuracy):
    plt.bar(classes, accuracy)
    plt.xticks(rotation=30, ha='right')
    plt.show()


def conf_matrix1(y_true, y_pred, classes, labels, ymap=None, figsize=(10,10)):

    print(confusion_matrix(y_true, y_pred))

    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=classes, columns=classes)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="Blues")
    plt.savefig('matrix.png')
    plt.show()

def conf_matrix2(y_true, y_pred, classes, labels, cmap=plt.cm.Blues):

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.set_printoptions(precision=2)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, rotation=45, ha='right')
    plt.yticks(np.arange(len(classes)+2))
    ax = plt.gca()

    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()


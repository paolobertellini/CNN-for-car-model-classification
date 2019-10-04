
import matplotlib.patches as mpatches
from pathlib import Path
from dataset import importMeta
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def printPlots(id, classes, dataset_dir, epochs, train_loss, train_acc, test_loss, test_acc, predict, true):

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

    lossHistory(id, 'Train', train_loss)
    accuracyHistory(id, 'Train', train_acc)

    lossHistory(id, 'Test', test_loss)
    accuracyHistory(id, 'Test', test_acc)

    #netAccuracy(classes, class_acc)
    lossComparison(id, train_loss, test_loss, epochs)
    accComparison(id, train_acc, test_acc, epochs)

    conf_matrix1(id, true, predict, classes, list(i for i in range(10)), figsize=(10, 10))
    conf_matrix2(id, true, predict, classes, list(i for i in range(10)))


def datasetDistribution(model_name, labels, classes, title):
    plt.bar(classes, labels)
    plt.xticks(rotation=30, ha='right')
    plt.title(f"{title} distribution")
    plt.savefig('plots/'+ model_name + '_' + 'e_dataset_distribution.png')
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
    plt.xticks(rotation=90, ha='right')
    plt.legend()
    plt.savefig('plots/dataset_comparison')
    plt.show()


def lossHistory(id, name, losses):
    plt.plot(losses, linewidth=4)
    plt.title(name + ' loss history', fontweight='bold')
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.savefig('plots/' + id + '_loss_' + name +'.png')
    plt.show()

def lossComparison(id, train_loss, test_loss, epochs):
    trainig = mpatches.Patch(color='royalblue', label='Training')
    testing = mpatches.Patch(color='orange', label='Testing')
    plt.plot(np.arange(epochs), train_loss, np.arange(epochs), test_loss, linewidth=4)
    plt.title('Loss comparison', fontweight='bold')
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.legend(handles=[trainig, testing])
    plt.savefig('plots/' + id + '_loss_comparison.png')
    plt.show()

def accComparison(id, train_acc, test_acc, epochs):
    trainig = mpatches.Patch(color='royalblue', label='Training')
    testing = mpatches.Patch(color='orange', label='Testing')
    plt.plot(np.arange(epochs), train_acc, np.arange(epochs), test_acc, linewidth=4)
    plt.title('Accuracy comparison', fontweight='bold')
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.ylim(0, 100)
    plt.legend(handles=[trainig, testing])
    plt.savefig('plots/' + id + '_acc_comparison.png')
    plt.show()

def accuracyHistory(id, name, accs):
    plt.plot(accs, linewidth=4)
    plt.title(name + ' accuracy history', fontweight='bold')
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.savefig('plots/' + id + '_acc_' + name +'.png')
    plt.show()


def conf_matrix1(id, y_true, y_pred, classes, labels, ymap=None, figsize=(10,10)):

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
    plt.savefig('lots/' + id + '_matrix.png')
    plt.show()

def conf_matrix2(id, y_true, y_pred, classes, labels, cmap=plt.cm.Blues):

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


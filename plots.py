
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataset import importMeta
import numpy as np

def printPlots(classes, dataset_dir, epochs, train_loss, train_acc, test_loss, test_acc, class_acc):

    train_labels = importMeta(dataset_dir / 'train')
    train_classes = list(0. for i in range(10))
    for l in train_labels:
        train_classes[l] += 1

    test_labels = importMeta(dataset_dir / 'test')
    test_classes = list(0. for i in range(10))
    for l in test_labels:
        test_classes[l] += 1

    datasetDistribution(train_classes, classes, 'Trainset')
    datasetDistribution(test_classes, classes, 'Testset')
    datasetComparison(classes, train_classes, test_classes)

    lossHistory('Train', train_loss)
    accuracyHistory('Train', train_acc)

    lossHistory('Test', test_loss)
    accuracyHistory('Test', test_acc)

    #netAccuracy(classes, class_acc)
    lossComparison(train_loss, test_loss, epochs)
    accComparison(train_acc, test_acc, epochs)

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
    trainig = mpatches.Patch(color='orange', label='Training')
    testing = mpatches.Patch(color='royalblue', label='Testing')
    plt.plot(np.arange(epochs), train_loss, np.arange(epochs), test_loss, linewidth=4)
    plt.title('Loss comparison', fontweight='bold')
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.legend(handles=[trainig, testing])
    plt.savefig('loss_comparison.png')
    plt.show()

def accComparison(train_acc, test_acc, epochs):
    trainig = mpatches.Patch(color='orange', label='Training')
    testing = mpatches.Patch(color='royalblue', label='Testing')
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
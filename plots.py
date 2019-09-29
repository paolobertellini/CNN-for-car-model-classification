
import matplotlib.pyplot as plt
from dataset import importMeta


def printPlots(classes, dataset_dir, losses, accs_train, accs_test):

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
    #datasetComparison(classes, train_classes, test_classes)

    lossHistory(losses)
    accuracyHistory(accs_train)
    netAccuracy(classes, accs_test)

def datasetDistribution(labels, classes, title):
    plt.bar(classes, labels)
    plt.xticks(rotation=30, ha='right')
    plt.title(f"{title} distribution")
    plt.show()

def datasetComparison(classes, train, test):
    plt.bar(classes, [train, test])
    plt.xticks(rotation=30, ha='right')
    plt.title('Trainset and testset distribution')
    plt.show()

def lossHistory(losses):
    plt.plot(losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

def accuracyHistory(accs):
    plt.plot(accs)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()

def netAccuracy(classes, accuracy):
    plt.bar(classes, accuracy)
    plt.xticks(rotation=30, ha='right')
    plt.show()
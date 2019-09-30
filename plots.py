
import matplotlib.pyplot as plt
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

    lossHistory(train_loss)
    accuracyHistory(train_acc)

    lossHistory(test_loss)
    accuracyHistory(test_acc)

    netAccuracy(classes, class_acc)
    lossComparison(train_loss, test_loss, epochs)

def datasetDistribution(labels, classes, title):
    plt.bar(classes, labels)
    plt.xticks(rotation=30, ha='right')
    plt.title(f"{title} distribution")
    plt.imsave()

def datasetComparison(classes, train, test):
    plt.bar(classes, train)
    plt.bar(classes, test)
    plt.xticks(rotation=30, ha='right')
    plt.title('Trainset and testset distribution')
    plt.show()

def lossHistory(losses):
    plt.plot(losses)
    plt.title('Training loss history')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def lossComparison(train_loss, test_loss, epochs):
    plt.plot(np.arange(epochs), train_loss, np.arange(epochs), test_loss)
    plt.show()

def accComparison(train_acc, test_acc, epochs):
    plt.plot(np.arange(epochs), train_acc, np.arange(epochs), test_acc)
    plt.show()

def accuracyHistory(accs):
    plt.plot(accs)
    plt.title('Training accuracy history')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

def netAccuracy(classes, accuracy):
    plt.bar(classes, accuracy)
    plt.xticks(rotation=30, ha='right')
    plt.show()
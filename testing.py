
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.optim as optim



def test(epochs, testloader, classes, batch_size, net, device):

    total_losses = []
    total_accs = []
    total_items = list(1. for i in range(10))
    total_corrects = list(1. for i in range(10))
    true = []
    predict = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        epoch_items = list(0. for i in range(10))
        epoch_corrects = list(0. for i in range(10))
        epoch_losses = list(0. for i in range(10))


        for i, batch in tqdm(enumerate(testloader), total=len(testloader)):

            images = batch['image']
            images = images.float().to(device)
            labels = batch['idx'].to(device)

            outputs = net(images)
            _, prediction = torch.max(outputs.data, 1)

            for item in range(batch_size):
                label = labels[item]
                epoch_items[label] += 1
                total_items[label] += 1
                predicted = prediction.data[item]
                predict.append(predicted.item())
                true.append(label.item())
                if (predicted == label):
                    epoch_corrects[label] += 1
                    total_corrects[label] += 1

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        epoch_avg_loss = np.asarray(epoch_losses).mean()
        epoch_acc = 100 * sum(epoch_corrects) / sum(epoch_items)

        total_losses.append(epoch_avg_loss)
        total_accs.append(epoch_acc)

        print(f'Testing epoch {epoch} statistics: [AVG LOSS: {epoch_avg_loss:.4f}, ACC: {epoch_acc:.2f}%]')

    print(f"Accuracy of the network on the test images: {100 * sum(total_corrects) / sum(total_items)}%")

    print(f"Class accuracy: ")
    class_accuracy = []
    for i in range(10):
        class_accuracy.append(100 * total_corrects[i] / (total_items[i]+1))
        print(f"[{classes[i]}: {class_accuracy[i]:.2f}%]")

    return total_losses, total_accs, class_accuracy, total_corrects, predict, true
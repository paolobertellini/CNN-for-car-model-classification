
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch

def train(epochs, batch_size,  trainloader, net, device):

    losses = []
    accs = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        items = 0
        corrects = 0
        avg_loss = []
        for batch in tqdm(trainloader, total=len(trainloader)):

            inputs = batch['image']
            inputs = inputs.float().to(device)
            labels = batch['idx'].to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            for item in range(batch_size):
                label = labels[item]
                items += 1
                if (predicted.data.cpu()[item] == label):
                    corrects += 1
            optimizer.zero_grad()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())

        avg_loss = np.asarray(avg_loss).mean()
        acc = 100 * corrects / items
        losses.append(avg_loss)
        accs.append(acc)
        print(f'Epoch {epoch} statistics: [AVG LOSS: {avg_loss:.4f}, ACC: {acc}%]')

    plt.plot(losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    plt.plot(accs)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()

    return losses, accs
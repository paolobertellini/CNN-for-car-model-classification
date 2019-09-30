
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch

def train(epochs, batch_size,  trainloader, net, device):

    total_losses = []
    total_accs = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        epoch_items = 0
        epoch_corrects = 0
        epoch_losses = []

        for batch in tqdm(trainloader, total=len(trainloader)):

            inputs = batch['image']
            inputs = inputs.float().to(device)
            labels = batch['idx'].to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            for item in range(batch_size):
                label = labels[item]
                epoch_items += 1
                if (predicted.data[item] == label):
                    epoch_corrects += 1

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        epoch_avg_loss = np.asarray(epoch_losses).mean()
        epoch_acc = 100 * epoch_corrects / epoch_items
        total_losses.append(epoch_avg_loss)
        total_accs.append(epoch_acc)
        print(f'Training epoch {epoch} statistics: [AVG LOSS: {epoch_avg_loss:.4f}, ACC: {epoch_acc:.2f}%]')

    return total_losses, total_accs

import matplotlib as plt
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train(epochs, trainloader, net, device):

    losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):

        avg_loss = []
        for batch in tqdm(trainloader, total=len(trainloader)):

            inputs = batch['image']
            inputs = inputs.float().to(device)
            labels = batch['idx'].to(device)

            outputs = net(inputs)
            optimizer.zero_grad()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())

        avg_loss = np.asarray(avg_loss).mean()
        losses.append(avg_loss)
        print(f'Average loss for epoch {epoch}: {avg_loss:.3f}')

    plt.plot(losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    return losses
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CarDataset
from model import SimpleCNN


plt.interactive(False)


def main(args):

    classes = ('Full size car', 'Mid size car', 'Cross over', 'Van',
               'Coupe', 'Family car', 'Beetle', 'Single seater', 'City car', 'Pick up')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.6242, 0.6232, 0.5952), (0.8963, 0.8787, 0.8833))])

    # -- DATASET -- #
    car_trainset = CarDataset(dataset_dir=args.dataset_dir / 'train',
                              transform=transform)
    car_trainloader = DataLoader(car_trainset, batch_size=4, shuffle=True)

    car_testset = CarDataset(dataset_dir=args.dataset_dir / 'test',
                             transform=transform)
    car_testloader = DataLoader(car_testset, batch_size=4, shuffle=True)

    # -- CONV NET -- #
    net = SimpleCNN().to(args.device)
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # -- TRAINING -- #
    for epoch in range(args.epochs):

        avg_loss = []

        for batch in tqdm(car_trainloader, total=len(car_trainloader)):

            inputs = batch['image']
            inputs = inputs.float().to(args.device)

            outputs = net(inputs)

            optimizer.zero_grad()

            labels = batch['idx'].to(args.device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            avg_loss.append(loss.item())

        avg_loss = np.asarray(avg_loss).mean()
        print(f'Average loss for epoch {epoch}: {avg_loss:.3f}')

    print('Finished Training')

    # -- TESTING -- #
    for i, batch in enumerate(car_testloader):
        labels, images = batch['idx'], batch['image']

    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    plt.show()
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in car_testloader:
            images, labels = batch['image'], batch['idx']
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for batch in car_testloader:
            images, labels = batch['image'], batch['idx']
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=Path)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()

    main(args)

import argparse
from pathlib import Path


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CarDataset, datasetHistogram
from model import SimpleCNN, CNN


plt.interactive(True)
plt.show(block=True)


def main(args):

    classes = ('Full size car', 'Mid size car', 'Cross over', 'Van',
               'Coupe', 'Family car', 'Beetle', 'Single seater', 'City car', 'Pick up')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.6242, 0.6232, 0.5952), (0.8963, 0.8787, 0.8833))])

    # -- DATASET -- #
    trainset = CarDataset(dataset_dir=args.dataset_dir / 'train',
                              transform=transform)
    #datasetHistogram(trainset.labels)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    testset = CarDataset(dataset_dir=args.dataset_dir / 'test',
                             transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)


    # -- CONV NET -- #
    net = CNN().to(args.device)
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    loss_avg = 0
    count = 0

    # -- TRAINING -- #

    losses = []

    for epoch in range(args.epochs):

        avg_loss = []

        for batch in tqdm(trainloader, total=len(trainloader)):

            inputs = batch['image']
            inputs = inputs.float().to(args.device)
            labels = batch['idx'].to(args.device)

            outputs = net(inputs)
            optimizer.zero_grad()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())

        avg_loss = np.asarray(avg_loss).mean()
        losses.append(avg_loss)
        plt.plot(epoch, avg_loss)
        plt.show()
        print(f'Average loss for epoch {epoch}: {avg_loss:.3f}')

    print('Finished Training, losses:', losses)



    # -- TESTING -- #

    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(testloader), total=len(testloader)):
            images, labels = batch['image'], batch['idx']
            images = images.float().to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            # print("OUT DATA: ", torch.max(outputs.data, 1))
            # print("PREDICTED: ", _, predicted)
            # print("PREDICTED 2: ", predicted.data.cpu() )
            # print("LABELS: ", labels, labels.size(0))

            total += labels.size(0)
            for i in range(labels.size(0)):
                if (predicted.data.cpu()[i] == labels[i]):
                    correct += 1

            print(f"BATCH TEST {i}: predicted {predicted.data.cpu()} true {labels}")

    print(f"Accuracy of the network on the test images: {100 * correct / total}%")
    #
    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    # with torch.no_grad():
    #     for batch in tqdm(testloader, total=len(testloader)):
    #         images, labels = batch['image'], batch['idx']
    #         outputs = net(images)
    #         images = images.float().to(args.device)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(args.batch_size):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1
    #
    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=Path)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    main(args)
    # from PIL import Image
    # plt.imshow(Image.open(args.dataset_dir / 'test' / 'img' / '000000.png'))
    # plt.show(block=True)
    # print(Image.open(args.dataset_dir / 'test' / 'img' / '000000.png').size)
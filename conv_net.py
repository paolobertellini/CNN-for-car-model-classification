import argparse
from pathlib import Path


import matplotlib
from numpy.ma import copy

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import time
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

criterion = nn.CrossEntropyLoss()

def train_model(model, trainloader, criteria, optimizer, scheduler,
                num_epochs=25, device='cuda'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch in trainloader:
                inputs = batch['image'].to(device)
                labels = batch['idx'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainloader.len()
            epoch_acc = running_corrects.double() / trainloader.len()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



    # -- TESTING -- #

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(testloader, total=len(testloader)):
            images, labels = batch['image'], batch['idx']
            images = images.float().to(args.device)
            outputs = net(images)
            print("")
            print("OUT DATA: ", outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            print("PREDICTED: ", _, predicted)
            print("LABELS: ", labels)
            total += labels.size(0)
            if (_, predicted == labels):
                correct += 1

    print(f"Accuracy of the network on the test images: {100 * correct / total}%")

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for batch in tqdm(testloader, total=len(testloader)):
            images, labels = batch['image'], batch['idx']
            outputs = net(images)
            images = images.float().to(args.device)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(args.batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=Path)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    main(args)
    # from PIL import Image
    # plt.imshow(Image.open(args.dataset_dir / 'test' / 'img' / '000000.png'))
    # plt.show(block=True)
    # print(Image.open(args.dataset_dir / 'test' / 'img' / '000000.png').size)
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CarDataset, datasetHistogram
from model import CNN
from training import train



def main(args):

    classes = ('Full size car', 'Mid size car', 'Cross over', 'Van',
               'Coupe', 'Family car', 'Beetle', 'Single seater', 'City car', 'Pick up')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.6242, 0.6232, 0.5952), (0.8963, 0.8787, 0.8833))])

    # -- DATASET -- #

    print(f"IMPORTING DATASET...")

    trainset = CarDataset(dataset_dir=args.dataset_dir / 'train',
                              transform=transform)
    datasetHistogram(trainset.labels)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    testset = CarDataset(dataset_dir=args.dataset_dir / 'test',
                             transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)


    # -- CONV NET -- #
    net = CNN().to(args.device)

    # -- TRAINING -- #
    print(f"STARTING TRAINING... ({args.epochs} epochs)")
    losses = train(args.epochs, trainloader, net, args.device)
    print("TRAINING FINISHED")
    print(f"Loss history: {losses}")

    # -- TESTING -- #
    correct = list(0. for i in range(10))
    total = list(0. for i in range(10))

    with torch.no_grad():
        for i, batch in tqdm(enumerate(testloader), total=len(testloader)):
            images, labels = batch['image'], batch['idx']
            images = images.float().to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            for item in range(args.batch_size):
                label = labels[item]
                total[label] += 1
                if (predicted.data.cpu()[item] == label):
                    correct[label] += 1

    tot_correct = sum(correct)
    tot = sum(total)

    print(f"Accuracy of the network on the test images: {100 * tot_correct / tot}%")
    accuracy = []
    for i in range(10):
        accuracy.append(100 * correct[i] / total[i])
        print('Accuracy of %5s class: %2d %%' % (
            classes[i], accuracy[i]))

    plt.bar(classes, accuracy)
    plt.show()
    #
    # correct = list(0. for i in range(10))
    # total = list(0. for i in range(10))
    # with torch.no_grad():
    #     for batch in tqdm(testloader, total=len(testloader)):
    #         images, labels = batch['image'], batch['idx']
    #         outputs = net(images)
    #         images = images.float().to(args.device)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(args.batch_size):
    #             label = labels[i]
    #             correct[label] += c[i].item()
    #             total[label] += 1
    #
    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * correct[i] / total[i]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=Path)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    main(args)


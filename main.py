import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CarDataset, datasetHistogram
from model import CNN
from training import train
from testing import test

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

    print(f"DATASET IMPORTED")

    # -- CONV NET -- #
    net = CNN().to(args.device)

    # -- TRAINING -- #
    print(f"STARTING TRAINING... ({args.epochs} epochs)")
    losses, accs = train(args.epochs, args.batch_size, trainloader, net, args.device)
    print("TRAINING FINISHED")
    print(f"Loss history: {losses}")
    print(f"Accuracy history: {accs}")

    # -- TESTING -- #
    print(f"STARTING TESTING... ({len(trainset)} images)")
    test(testloader, classes, args.batch_size, net, args.device)
    print("TESTING FINISHED")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=Path)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    main(args)


import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from dataset import CarDataset
from finetuning import finetuning
from model import CNN
from training import train
from testing import test
import plots


#from finetunig import initialize_model

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}..")

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
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    testset = CarDataset(dataset_dir=args.dataset_dir / 'test',
                             transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    print(f"DATASET IMPORTED")

    # -- CONV NET -- #
    net = CNN().to(device)

    # statistics
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    true_list = []
    predict_list = []

    for epoch in range(args.epochs):

        # taining
        train_loss, train_acc = train(trainloader, net, args.batch_size, args.learning_rate, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        #test
        test_loss, test_acc, true, predict = test(args.batch_size, testloader, net, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        true_list += true
        predict_list += predict

        print(f"EPOCH {epoch}: [TRAINING loss: {train_loss:.5f} acc: {train_acc:.2f}%]",
                             f"[TESTING loss: {test_loss:.5f} acc: {test_acc:.2f} %]")


    print(f"TRAINING AND TESTING FINISHED")
    print(f"TRAIN LOSS HISTORY: {train_losses}")
    print(f"TRAIN ACCURACY HISTORY: {train_accs}")
    print(f"TEST LOSS HISTORY: {test_losses}")
    print(f"TEST ACCURACY HISTORY: {test_accs}")
    plots.printPlots(classes, args.dataset_dir, args.epochs, train_losses, train_accs, test_losses, test_accs, predict_list, true_list)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=Path)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0001)

    #parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(args)
    #finetuning(args.dataset_dir)

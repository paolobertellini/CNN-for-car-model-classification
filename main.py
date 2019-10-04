
import argparse
from pathlib import Path
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import plots
from dataset import CarDataset
from finetuning import initialize_model
from testing import test
from training import train

def read(filename):
    list = []
    with open('data/' + filename, 'rt')as f:
        data = csv.reader(f)
        for row in data:
            list.append(float(row[0]))
    return list

def write(filename, list):
    with open('data/' + filename, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in list:
            writer.writerow([val])

def main(args):

    # conv net model
    # Models to choose from [mini, paolo, resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "vgg"

    # hyperparameters
    num_classes = 10
    batch_size = 4
    feature_extract = True
    use_pretrained = True

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}..")

    # vehicles classes
    classes = ('Full size car', 'Mid size car', 'Cross over', 'Van',
               'Coupe', 'Family car', 'Beetle', 'Single seater', 'City car', 'Pick up')

    # model architecture
    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained)
    model = model\
        .to(device)
    print('-' * 100)
    print(f"MODEL ARCHITECTURE [{model_name}]")
    print('-' * 100)
    print(model)

    # params
    params_to_update = model.parameters()
    print('-' * 100)
    print("PARAMS TO LEARN")
    print('-' * 100)
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # image transformations
    transform2 = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.6242, 0.6232, 0.5952), (0.8963, 0.8787, 0.8833))])

    transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # dataset
    print('-' * 100)
    print(f"IMPORTING DATASET...")
    print('-' * 100)

    trainset = CarDataset(dataset_dir=args.dataset_dir / 'train',
                          transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    testset = CarDataset(dataset_dir=args.dataset_dir / 'test',
                         transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    print('-' * 100)
    print(f"DATASET IMPORTED")
    print('-' * 100)

    # statistics
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    true_list = []
    predict_list = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    for epoch in range(args.epochs):

        print(f"EPOCH {epoch+1}/{args.epochs}")
        # taining
        print("Training")
        train_loss, train_acc = train(trainloader, model, args.batch_size, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # test
        print("Testing")
        test_loss, test_acc, true, predict = test(testloader, model, args.batch_size, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        true_list += true
        predict_list += predict

        print(f"EPOCH {epoch+1}: [TRAINING loss: {train_loss:.5f} acc: {train_acc:.2f}%]",
              f"[TESTING loss: {test_loss:.5f} acc: {test_acc:.2f} %]")
        print('-' * 100)

    print('-' * 100)
    print(f"TRAINING AND TESTING FINISHED")
    print('-' * 100)
    print(f"TRAIN LOSS HISTORY: {train_losses}")
    write('train_loss.csv', train_losses)
    p = read('train_loss.csv')

    print(f"TRAIN ACCURACY HISTORY: {train_accs}")
    print('-' * 100)
    print(f"TEST LOSS HISTORY: {test_losses}")
    print(f"TEST ACCURACY HISTORY: {test_accs}")
    print('-' * 100)
    plots.printPlots(classes, args.dataset_dir, args.epochs, p, train_accs, test_losses, test_accs,
                     predict_list, true_list)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=Path)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0001)

    #parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()

    main(args)

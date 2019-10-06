
import argparse
from pathlib import Path
import csv

import torch
import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import plots
from dataset import CarDataset
from finetuning import initialize_model
from testing import test
from training import train
import main


def execute(device, model_name, dataset_dir, batch_size, epochs, learning_rate, num_classes, feature_extract, use_pretrained, save_file, print_plots, finetuning):

    # vehicles classes
    classes = ('Full size car', 'Mid size car', 'Cross over', 'Van',
               'Coupe', 'Family car', 'Beetle', 'Single seater', 'City car', 'Pick up')

    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained)
    model = model.to(device)
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

    trainset = CarDataset(dataset_dir=dataset_dir / 'train',
                          transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = CarDataset(dataset_dir=dataset_dir / 'test',
                         transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

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
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    criterion2 = nn.CrossEntropyLoss()
    optimizer2 = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(epochs):
        print(f"EPOCH {epoch + 1}/{epochs}")
        # taining
        print("Training")
        train_loss, train_acc = train(trainloader, model, batch_size, criterion, optimizer, finetuning, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # test
        print("Testing")
        test_loss, test_acc, true, predict = test(testloader, model, batch_size, criterion2, finetuning, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        true_list += true
        predict_list += predict

        print(f"EPOCH {epoch + 1}: [TRAINING loss: {train_loss:.5f} acc: {train_acc:.2f}%]"
              ,f"[TESTING loss: {test_loss:.5f} acc: {test_acc:.2f}%]")
        print('-' * 100)

    print('-' * 100)
    print(f"TRAINING AND TESTING FINISHED")
    print('-' * 100)

    # train
    print(f"TRAIN LOSS HISTORY: {train_losses}")
    print('-' * 100)
    print(f"TRAIN ACCURACY HISTORY: {train_accs}")
    print('-' * 100)

    # # test
    # print('-' * 100)
    # print(f"TEST LOSS HISTORY: {test_losses}")
    # print('-' * 100)
    # print(f"TEST ACCURACY HISTORY: {test_accs}")
    # print('-' * 100)
    # print('-' * 100)

    d = datetime.datetime.now()
    date_id = str(getattr(d, 'year')) + '-' + str(getattr(d, 'month')) + '-' + str(getattr(d, 'day')) + '__' + str(
        getattr(d, 'hour')) + ':' + str(getattr(d, 'minute'))
    id = model_name + '__' + date_id + '__' + str(epochs) + 'e__'

    if save_file:
        print("Saving files...")
        id_save = 'data/' + id
        main.write('data/' + id + 'train_loss.csv', train_losses)
        main.write('data/' + id + 'train_acc.csv', train_accs)
        main.write('data/' + id + 'test_loss.csv', test_losses)
        main.write('data/' + id + 'test_acc.csv', test_accs)
        print("FILES SAVED")
        print('-' * 100)

    if print_plots:
        print("Printing plots...")
        plots.printPlots(id, classes, dataset_dir, epochs, train_losses, train_accs, test_losses, test_accs,
                         predict_list, true_list)
        print("PLOTS PRINTED AND SAVED")
        print('-' * 100)

    print('-' * 100)
    print('-' * 100)
    print("FINISH")
    print('-' * 100)
    print('-' * 100)


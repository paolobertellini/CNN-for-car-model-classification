import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import main
import plots
from dataset import CarDataset
from finetuning import initialize_model
from plots import conf_matrix1
from testing import test
from training import train


def execute(device, model_name, dataset_dir, batch_size, epochs, learning_rate, num_classes, feature_extract, use_pretrained, save_file, print_plots):

    # vehicles classes
    classes = ('Full size car', 'Mid size car', 'Cross over', 'Van',
               'Coupe', 'Family car', 'Beetle', 'Single seater', 'City car', 'Pick up')

    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained)
    model = model.to(device)
    print('-' * 100)
    print(f"MODEL ARCHITECTURE [{model_name}] [feature extract: {feature_extract}, use_pretrained: {use_pretrained} learning rate: {learning_rate}]")
    # print('-' * 100)
    # print(model)

    # params
    params_to_update = model.parameters()
    print('-' * 100)
    print("PARAMS TO LEARN")
    print('-' * 100)

    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("\t", name)

    transform_dict = {
        'train': transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'eval': transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    }

    # dataset
    print('-' * 100)
    print(f"IMPORTING DATASET...")
    print('-' * 100)

    trainset = CarDataset(dataset_dir=dataset_dir / 'train',
                          transform=transform_dict['train'])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

    testset = CarDataset(dataset_dir=dataset_dir / 'test',
                         transform=transform_dict['eval'])
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True)

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

    optimizer = optim.Adam(params_to_update, lr=learning_rate)

    d = datetime.datetime.now()
    date_id = str(getattr(d, 'month')) + '-' + str(getattr(d, 'day')) + '__' + str(
        getattr(d, 'hour')) + ':' + str(getattr(d, 'minute'))
    id = model_name + '__' + date_id + '__' + str(learning_rate) + 'e__'
    if feature_extract:
        id += 'FE__'

    best_test_acc = 0.

    for epoch in range(1, epochs):
        print(f"EPOCH {epoch}/{epochs}")

        # training
        print("Training")
        train_loss, train_acc = train(trainloader, model, batch_size, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"EPOCH {epoch + 1}: [TRAINING loss: {train_loss:.5f} acc: {train_acc:.2f}%]")

        # test
        print("Testing")

        with torch.no_grad():
            test_loss, test_acc, true, predict = test(testloader, model, batch_size, criterion, device)

        if test_acc > best_test_acc:

            best_test_acc = test_acc

            # Keep track of best epoch
            best_epoch_file = Path('data') / (id + 'best_epoch.txt')
            with best_epoch_file.open('wt') as f:
                f.write(str(epoch))

            # Save best model weights
            torch.save(model.state_dict(), 'data/' + id + 'weights.pth')

            conf_matrix1(id, true, predict, classes, list(i for i in range(10)), figsize=(10, 10))

        test_losses.append(test_loss)
        test_accs.append(test_acc)
        true_list += true
        predict_list += predict
        print(f"EPOCH {epoch}: [TESTING loss: {test_loss:.5f} acc: {test_acc:.2f}%]")

        print('-' * 100)

        if save_file:
            main.write('data/' + id + 'train_loss.csv', train_losses)
            main.write('data/' + id + 'train_acc.csv', train_accs)
            main.write('data/' + id + 'test_loss.csv', test_losses)
            main.write('data/' + id + 'test_acc.csv', test_accs)
            main.write('data/' + id + 'true.csv', true_list)
            main.write('data/' + id + 'predict.csv', predict_list)

        if print_plots:
            plots.printPlots(id, classes, dataset_dir, epoch, train_losses, train_accs,
                             test_losses, test_accs,
                             predict_list, true_list)


    print('-' * 100)
    print(f"TRAINING AND TESTING FINISHED")
    print('-' * 100)

    # train
    print(f"TRAIN LOSS HISTORY: {train_losses}")
    print('-' * 100)
    print(f"TRAIN ACCURACY HISTORY: {train_accs}")
    print('-' * 100)

    # test
    print('-' * 100)
    print(f"TEST LOSS HISTORY: {test_losses}")
    print('-' * 100)
    print(f"TEST ACCURACY HISTORY: {test_accs}")
    print('-' * 100)
    print('-' * 100)

    print('-' * 100)
    print('-' * 100)
    print("FINISH")
    print('-' * 100)
    print('-' * 100)

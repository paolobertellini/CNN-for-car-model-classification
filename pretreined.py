
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from dataset import CarDataset
from training import train
from finetuning import initialize_model


def pretreined(args, device):
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]

    num_classes = 10
    batch_size = 4
    num_epochs = 2

    feature_extract = True

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    print(model_ft)
    transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    print("Initializing Datasets and Dataloaders...")

    trainset = CarDataset(dataset_dir=args.dataset_dir / 'train',
                          transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    model_ft = model_ft.to(device)

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Train and evaluate
    model_ft, hist = train(trainloader, model_ft, batch_size, 0.0001, device)

    # Initialize the non-pretrained version of the model used for this run
    # scratch_model, _ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
    # scratch_model = scratch_model.to(device)
    # scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
    # scratch_criterion = nn.CrossEntropyLoss()
    # _, scratch_hist = train(scratch_model, device, trainloader, scratch_criterion, scratch_optimizer,
    #                               num_epochs=num_epochs)
    #
    # ohist = [h.cpu().numpy() for h in hist]
    # shist = [h.cpu().numpy() for h in scratch_hist]
    #
    # plt.title("Validation Accuracy vs. Number of Training Epochs")
    # plt.xlabel("Training Epochs")
    # plt.ylabel("Validation Accuracy")
    # plt.plot(range(1, num_epochs + 1), ohist, label="Pretrained")
    # plt.plot(range(1, num_epochs + 1), shist, label="Scratch")
    # plt.ylim((0, 1.))
    # plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    # # plt.legend()
    # plt.show()
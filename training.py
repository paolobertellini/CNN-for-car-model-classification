
from tqdm import tqdm

import numpy as np
import torch
import time
import copy


def train(trainloader, net, batch_size, criterion, optimizer, device):

    #best_model_wts = copy.deepcopy(net.state_dict())


    #net.train()  # Set model to training mode

    epoch_items = 0
    epoch_corrects = 0
    epoch_losses = []

    for batch in tqdm(trainloader, total=len(trainloader)):

        inputs = batch['image']
        inputs = inputs.float().to(device)
        labels = batch['idx'].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        for item in range(batch_size):
            label = labels[item]
            epoch_items += 1
            if (predicted.data[item] == label):
                epoch_corrects += 1


        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # statistics
        epoch_losses.append(loss.item())

    epoch_avg_loss = np.asarray(epoch_losses).mean()
    epoch_acc = 100 * epoch_corrects / epoch_items
    #net.load_state_dict(best_model_wts)
    return epoch_avg_loss, epoch_acc

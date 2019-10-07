
import torch
from tqdm import tqdm
import numpy as np


def test(testloader, net, batch_size, criterion, device):

    true = []
    predict = []

    epoch_items = list(0. for i in range(10))
    epoch_corrects = list(0. for i in range(10))
    epoch_losses = list(0. for i in range(10))

    net.eval()

    for batch in tqdm(testloader, total=len(testloader)):

        images = batch['image']
        images = images.float().to(device)
        labels = batch['idx'].to(device)

        outputs = net(images)
        _, prediction = torch.max(outputs.data, 1)

        for item in range(batch_size):
            label = labels[item]
            epoch_items[label] += 1

            predicted = prediction.data[item]
            predict.append(predicted.item())
            true.append(label.item())

            if predicted == label:
                epoch_corrects[label] += 1

        loss = criterion(outputs, labels)
        epoch_losses.append(loss.item())

    epoch_avg_loss = np.asarray(epoch_losses).mean()
    epoch_acc = 100 * sum(epoch_corrects) / sum(epoch_items)

    return epoch_avg_loss, epoch_acc, true, predict

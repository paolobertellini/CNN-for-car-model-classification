
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def test(testloader, classes, batch_size, net, device):

    correct = list(0. for i in range(10))
    total = list(0. for i in range(10))

    with torch.no_grad():
        for i, batch in tqdm(enumerate(testloader), total=len(testloader)):
            images, labels = batch['image'], batch['idx']
            images = images.float().to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            for item in range(batch_size):
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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CarDataset
from model import SimpleCNN
from pathlib import Path
import argparse


plt.interactive(False)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()


def main(args):

    classes = ('Full size car', 'Mid size car', 'Cross over', 'Van',
               'Coupe', 'Family car', 'Beetle', 'Single seater', 'City car', 'Pick up')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.6242, 0.6232, 0.5952), (0.8963, 0.8787, 0.8833))])

    # -- DATASET -- #
    car_trainset = CarDataset(dataset_dir=args.dataset_dir / 'train',
                              transform=transform)
    print(f'Train set: {len(car_trainset)}')

    # for i in range(len(car_trainset)):
    #     data = car_trainset[i]
    #     print('Dataset TRAIN element', i,':', data['idx'], data['image'].shape)
    #     imshow(data['image'])
    #     plt.show()

    car_trainloader = DataLoader(car_trainset, batch_size=4,
                                 shuffle=True, num_workers=0)
    train_iter = iter(car_trainloader)

    # for i, data in enumerate(train_iter):
    #     print('Batch TRAIN ', i, 'train loader:', data['idx'], data['image'].shape)

    car_testset = CarDataset(dataset_dir=args.dataset_dir / 'test',
                             transform=transform)

    # for i in range(len(car_testset)):
    #     data = car_trainset[i]
    #     print('Dataset TEST element', i,':', data['idx'], data['image'].shape)

    car_testloader = DataLoader(car_testset, batch_size=4,
                                shuffle=True, num_workers=0)
    test_iter = iter(car_testloader)

    # for i, data in enumerate(test_iter):
    #     print('Batch TEST ', i, 'train loader:', data['idx'], data['image'].shape)

    # show images
    # imshow(torchvision.utils.make_grid(images))
    # plt.show()

    # -- CONV NET -- #
    net = SimpleCNN()
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # -- TRAINING -- #

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(car_trainloader, 0):
            # get the inputs
            inputs = data['image']
            inputs = inputs.float()
            labels = data['idx']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1900:
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # -- TESTING -- #

    test_iter = iter(car_testloader)
    for i, batch in enumerate(test_iter):
        labels, images = batch['idx'], batch['image']

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    plt.show()
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in car_testloader:
            images, labels = data['image'], data['idx']
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in car_testloader:
            images, labels = data['image'], data['idx']
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=Path)
    args = parser.parse_args()

    main(args)



'''
MEDIUM

GroundTruth:  Cross over Pick up Mid size car Family car
Predicted:  Family car Family car Family car City car
Accuracy of the network on the 10000 test images: 30 %
Accuracy of Full size car :  6 %
Accuracy of Mid size car :  0 %
Accuracy of Cross over : 10 %
Accuracy of   Van :  0 %
Accuracy of Coupe :  0 %
Accuracy of Family car : 80 %
Accuracy of Beetle :  0 %
Accuracy of Single seater :  0 %
Accuracy of City car : 39 %
Accuracy of Pick up : 23 %
'''
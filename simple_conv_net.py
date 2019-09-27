import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import glob
import yaml
from PIL import Image
from skimage import transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

plt.interactive(False)

dir = 'C:\car\complete'

classes = ('Full size car', 'Mid size car', 'Cross over', 'Van',
           'Coupe', 'Family car', 'Beetle', 'Single seater', 'City car', 'Pick up')


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        idx = sample['idx']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'idx': idx}


class ToTensor(object):

    def __call__(self, sample):
        image, idx = sample['image'], sample['idx']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'idx': idx}


class CarDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):

        return len(glob.glob1(self.root_dir + '\img', "*.png"))

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        name = ''
        for i in range(6 - len(str(idx))):
            name += '0'
        name += str(idx)
        img_name = name + '.png'
        meta_name = name + '.yaml'

        image = Image.open(self.root_dir + '/img/' + img_name)
        f = open(self.root_dir + '/meta/' + meta_name)
        dataMap = yaml.safe_load(f)
        cad_idx = dataMap["cad_idx"]
        sample = {'image': image, 'idx': cad_idx}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.6242, 0.6232, 0.5952), (0.8963, 0.8787, 0.8833))])


# -- DATASET -- #

car_trainset = CarDataset(root_dir=dir + '/train', transform=transform)

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

car_testset = CarDataset(root_dir=dir + '/test', transform=transform)

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

class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        # 64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 18 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
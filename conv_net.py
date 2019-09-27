import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml
from PIL import Image
from skimage import transform
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import dataset_statistics as ds

plt.interactive(False)

dir = 'C:\car\medium'

# Hyperparameters
num_epochs = 5
num_classes = 10
batch_size = 20
learning_rate = 0.001
momentum = 0.9

classes = ('Full size car', 'Mid size car', 'Cross over', 'Van',
           'Coupe', 'Family car', 'Beetle', 'Single seater', 'City car', 'Pick up')


class ToTensor(object):

    def __call__(self, sample):
        image, idx = sample['image'], sample['idx']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'idx': idx}


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


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.6242, 0.6232, 0.5952), (0.8963, 0.8787, 0.8833))])

# -- TRAIN set -- #

car_trainset = CarDataset(root_dir=dir + '/train', transform=transform)

# for i in range(len(car_trainset)):
#     data = car_trainset[i]
#     print('Dataset TRAIN element', i, ':', data['idx'], data['image'].shape)
#     #plt.show(data['image'])

car_trainloader = DataLoader(car_trainset, batch_size=batch_size,
                             shuffle=True, num_workers=0)


ds.calculate_img_stats_avg(car_trainloader)

train_iter = iter(car_trainloader)

# for i, data in enumerate(train_iter):
#     print('Batch TRAIN ', i, 'train loader:', data['idx'], data['image'].shape)


# -- TEST set -- #

car_testset = CarDataset(root_dir=dir + '/test', transform=transform)

# for i in range(len(car_testset)):
#     data = car_trainset[i]
#     print('Dataset TEST element', i, ':', data['idx'], data['image'].shape)

car_testloader = DataLoader(car_testset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
test_iter = iter(car_testloader)

# for i, data in enumerate(test_iter):
#     print('Batch TEST ', i, 'train loader:', data['idx'], data['image'].shape)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(car_trainloader)
data = dataiter.next()
images = data['image']
labels = data['idx']

# show images
imshow(torchvision.utils.make_grid(images))
plt.show()

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# Train the model
total_step = len(car_trainloader)
for epoch in range(num_epochs):
    for i, data in enumerate(car_trainloader):
        images = data['image']
        labels = data['idx']

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

outputs = model(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in car_testloader:
        images, labels = data['image'], data['idx']
        outputs = model(images)
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
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
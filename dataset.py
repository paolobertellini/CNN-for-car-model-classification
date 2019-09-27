import glob

import torch
import yaml
from PIL import Image
from skimage import transform
from torch.utils.data import Dataset
from pathlib import Path


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

    def __init__(self, dataset_dir: Path, transform=None):

        if not dataset_dir.is_dir():
            raise OSError(f'Folder {str(dataset_dir)} not found.')

        self.dataset_dir = dataset_dir
        self.transform = transform

    def __len__(self):
        return len(list((self.dataset_dir / 'img').glob('*.png')))

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        name = ''
        for i in range(6 - len(str(idx))):
            name += '0'
        name += str(idx)
        img_name = name + '.png'
        meta_name = name + '.yaml'

        image = Image.open(self.dataset_dir / 'img' / img_name)
        f = open(self.dataset_dir / 'meta' / meta_name)

        dataMap = yaml.safe_load(f)
        cad_idx = dataMap["cad_idx"]
        sample = {'image': image, 'idx': cad_idx}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

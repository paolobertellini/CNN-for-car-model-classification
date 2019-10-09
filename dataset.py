from pathlib import Path

import torch
import yaml
from PIL import Image
from skimage import transform
from torch.utils.data import Dataset


def import_meta(dataset_dir):
    labels = []
    for meta_f in list(sorted((dataset_dir / 'meta').glob('*.yaml'))):
        with meta_f.open('r') as f:
            cad_idx = yaml.safe_load(f)['cad_idx']
        labels.append(cad_idx)
    return labels


def calculate_img_stats_avg(loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in loader:
        imgs = data['image']
        batch_samples = imgs.size(0)
        imgs = imgs.view(batch_samples, imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print (mean, std)
    return mean, std


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
        self.labels = import_meta(self.dataset_dir)
        print(f"Imported labels from {self.dataset_dir}")

    def __len__(self):
        return len(list((self.dataset_dir / 'img').glob('*.png')))

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = f'{idx:06d}.png'

        image = Image.open(self.dataset_dir / 'img' / img_name)

        cad_idx = self.labels[idx]
        sample = {'image': image, 'idx': cad_idx}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

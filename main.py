
import os
os.environ['OMP_NUM_THREADS'] = "1"

import argparse
import csv
from pathlib import Path

import torch

import configuration
from plots import printPlotById


def read(filename):
    list = []
    with open('data/' + filename, 'rt')as f:
        data = csv.reader(f)
        for row in data:
            list.append(float(row[0]))
    return list


def write(filename, list):
    with open(filename, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in list:
            writer.writerow([val])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=Path)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0005)

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}..")

    # vehicles classes
    classes = ('Full size car', 'Mid size car', 'Cross over', 'Van',
               'Coupe', 'Family car', 'Beetle', 'Single seater', 'City car', 'Pick up')

    printPlotById('lenet5__10-11__14_21__0.001e__', args.dataset_dir, classes)

    # configuration.execute(device=device,
    #                       model_name='vgg19',
    #                       dataset_dir=args.dataset_dir,
    #                       batch_size=args.batch_size,
    #                       epochs=50,
    #                       learning_rate=0.0001,
    #                       num_classes=10,
    #                       feature_extract=False,
    #                       use_pretrained=True,
    #                       save_file=True,
    #                       print_plots=True)


import argparse
import csv
from pathlib import Path

import torch

import configuration


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
    parser.add_argument('--learning_rate', type=float, default=0.001)

    #parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}..")

    configuration.execute(device=device,
                          model_name='vgg',
                          dataset_dir=args.dataset_dir / 'complete',
                          batch_size=4,
                          epochs=101,
                          learning_rate=0.0001,
                          num_classes=10,
                          feature_extract=True,
                          use_pretrained=True,
                          save_file=True,
                          print_plots=True,
                          finetuning=True)
    #
    configuration.execute(device=device,
                          model_name='vgg',
                          dataset_dir=args.dataset_dir / 'complete',
                          batch_size=4,
                          epochs=100,
                          learning_rate=0.0001,
                          num_classes=10,
                          feature_extract=False,
                          use_pretrained=True,
                          save_file=True,
                          print_plots=True,
                          finetuning=False)

    # configuration.execute(device=device,
    #                       model_name='paolo',
    #                       dataset_dir=args.dataset_dir / 'complete',
    #                       batch_size=4,
    #                       epochs=300,
    #                       learning_rate=0.001,
    #                       num_classes=10,
    #                       feature_extract=False,
    #                       use_pretrained=True,
    #                       save_file=True,
    #                       print_plots=True,
    #                       finetuning=False)

    # configuration.execute(device=device,
    #                       model_name='vgg',
    #                       dataset_dir=args.dataset_dir / 'complete',
    #                       batch_size=4,
    #                       epochs=100,
    #                       learning_rate=0.0001,
    #                       num_classes=10,
    #                       feature_extract=False,
    #                       use_pretrained=True,
    #                       save_file=True,
    #                       print_plots=True,
    #                       finetuning=False)


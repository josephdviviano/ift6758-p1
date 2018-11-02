# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import torch
import torchvision
import torchvision.transforms as transforms

@click.command()
def main():
    """ Runs scripts to download raw data into (../raw) and processed data into (../processed)."""
    logger = logging.getLogger(__name__)
    logger.info('Downloading raw data')

    project_dir = Path(__file__).resolve().parents[2].as_posix()
    data_dir = os.path.join(project_dir, 'data')

    print('Generating preprocessed data...')

    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    train_data = torch.stack(tuple([i for i, _ in train_dataset]), 0)
    test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    test_data = torch.stack(tuple([i for i, _ in test_dataset]), 0)

    print('Done!')

    print('Removing ubyte files...')

    for file in os.listdir(os.path.join(data_dir, 'raw')):
        if not file.startswith('.'):
            os.remove(os.path.join(data_dir, 'raw', file))
    os.rename(os.path.join(data_dir, 'processed', 'training.pt'),
              os.path.join(data_dir, 'raw', 'training.pt'))
    os.rename(os.path.join(data_dir, 'processed', 'test.pt'),
              os.path.join(data_dir, 'raw', 'test.pt'))

    print('Done!')

    with open(os.path.join(data_dir, 'processed', 'training.pt'), 'wb') as f:
        torch.save((train_data, train_dataset.train_labels), f)
    with open(os.path.join(data_dir, 'processed', 'test.pt'), 'wb') as f:
        torch.save((test_data, test_dataset.test_labels), f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()

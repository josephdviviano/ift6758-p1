# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from models.lenet import LeNet


@click.command()
@click.option('-bs', '--batch-size', type=int, default=64,
              show_default=True, help='Input batch size for training')
@click.option('-tbs', '--test-batch-size', type=int, default=1000,
              show_default=True, help='Input batch size for testing')
@click.option('-e', '--epochs', type=int, default=10,
              show_default=True, help='Number of epochs to train')
@click.option('-lr', '--learning-rate', type=float, default=0.01,
              show_default=True, help='Learning rate')
@click.option('-c/-nc', '--cuda/--no-cuda', default=False,
              show_default=True, help='Enable CUDA training')
@click.option('-s', '--seed', type=int, default=1,
              show_default=True, help='Random seed')
@click.option('-li', '--log-interval', type=int, default=1,
              show_default=True, help='Batches to wait before logging training')

def main(batch_size, test_batch_size, epochs, lr, cuda, seed, log_interval):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    TRAIN_DATA = os.path.join(Path(__file__).resolve().parents[2], 'data', 'processed', 'training.pt')
    TEST_DATA = os.path.join(Path(__file__).resolve().parents[2], 'data', 'processed', 'test.pt')

    torch.manual_seed(seed)
    device = torch.device("cuda" if (cuda and torch.cuda.is_available()) else "cpu")

    x_train, y_train = torch.load(TRAIN_DATA)
    x_test, y_test = torch.load(TEST_DATA)

    train = data_utils.TensorDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
    test = data_utils.TensorDataset(x_test, y_test)
    test_loader = data_utils.DataLoader(test, batch_size=test_batch_size, shuffle=True)

    model = LeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

def train():
    pass

def test():
    pass


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
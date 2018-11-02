# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import torch
import torch.nn as nn
import torch.nn.functional as FN
import torchvision.models as models


@click.command()
@click.option('--path', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Path to file', show_default=True,
              default=os.path.join(Path(__file__).resolve().parents[2], 'data', 'processed'))
@click.option('--architecture', type=click.Choice(['vgg16', 'alexnet']), help='Architecture for feature extraction',
              show_default=True, default='vgg16')
def main(path, architecture):
    """ Runs the feature extraction to turn processed data from (../processed) into
        features that are saved (saved in ../features).
    """
    logger = logging.getLogger(__name__)
    logger.info('Extracting features from processed data')

    x_train, y_train = torch.load(os.path.join(path, 'training.pt'))
    # train_loader = torch.utils.data.TensorDataset(x_train, y_train)
    # train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=8, shuffle=True)
    x_test, y_test = torch.load(os.path.join(path, 'test.pt'))
    # test_loader = torch.utils.data.TensorDataset(x_test, y_test)
    # test_loader_dataset = torch.utils.data.DataLoader(test_loader, batch_size=8, shuffle=True)

    options = {'vgg16': vgg16, 'alexnet':alexnet}
    print(x_train.shape)
    print(x_test.shape)
    options[architecture](x_train[0:2, :, :, :])

def vgg16(x):
    #upsample = FN.interpolate(x, (32, 32))
    vgg16 = models.vgg16(pretrained=True)
    vgg16_features = Features(vgg16)
    output = vgg16_features.forward(x)
    print(output.shape)

def alexnet(x):
    #upsample = FN.interpolate(x, (64, 64))
    alexnet = models.alexnet(pretrained=True)
    vgg16_features = Features(alexnet)
    output = vgg16_features.forward(x)
    print(output.shape)


class Features(nn.Module):
    def __init__(self, original_model):
        super(Features, self).__init__()
        self.features = nn.Sequential(*list(original_model.features.children()))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return x


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

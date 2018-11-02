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
@click.option('--path', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Path to file', show_default=True,
              default=os.path.join(Path(__file__).resolve().parents[2], 'data', 'processed', 'training.pt'))
@click.option('--architecture', type=click.Choice(['vgg16', 'resnet']), help='Architecture for feature extraction',
              show_default=True, default='vgg16')
def main(path, architecture):
    """ Runs the feature extraction to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')
    print(path)
    print(architecture)
    x, y = torch.load(path)
    options = {'vgg16': vgg16}
    print(x.shape)
    #options[architecture](x)

def vgg16(x):
    FN.upsample()
    vgg16 = models.vgg16()
    vgg16_features = Features(vgg16, 2)
    output = vgg16_features(x)
    print(output.shape)


class Features(nn.Module):
    def __init__(self, original_model, layer):
        super(Features, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-layer])

    def forward(self, x):
        x = self.features(x)
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

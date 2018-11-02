import torch
from train_net import Net
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
    batch_size=1000, shuffle=False)


train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
    batch_size=1000, shuffle=False)

model = Net()
model.load_state_dict(torch.load('models/model.pt'))
model.eval()

vectors = np.empty((1, 50))
labels = np.empty(((1,)))

for data, target in tqdm(train_loader):
    vec = model(data, get_vectors=True).detach().numpy()
    vectors = np.append(vectors, vec, axis=0)
    labels = np.append(labels, target.detach().numpy(), axis=0)

for data, target in tqdm(test_loader):
    vec = model(data, get_vectors=True).detach().numpy()
    vectors = np.append(vectors, vec, axis=0)
    labels = np.append(labels, target.detach().numpy(), axis=0)

vectors = np.delete(vectors, (0), axis=0)
labels = np.delete(labels, (0), axis=0)

to_save = {'data': vectors, 'labels': labels}
np.save('data/vectors.npy', to_save)
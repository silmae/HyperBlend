
##########################
# From tutorial: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
##########################

import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import save
from torch import load
from torch import from_numpy
import torch.optim as optim

import src.leaf_model.training_data
from src.data import path_handling as PH
from src.data import toml_handling as TH
from src import constants as C
from src import plotter
from src.leaf_model import leaf_commons as shared


torch.manual_seed(666)


class Net(nn.Module):

    def __init__(self, layer_count, layer_width):
        super(Net, self).__init__()
        input_dim = 2
        output_dim = 4
        self.layer_count = layer_count
        self.layer_width = layer_width
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for i in range(layer_count+1):
            layer = nn.Linear(current_dim, layer_width)
            # nn.init.constant_(layer.weight, val=1.0)#nonlinearity='relu')
            self.layers.append(layer)
            current_dim = layer_width
        self.layers.append(nn.Linear(current_dim, output_dim))
        self.activation = F.leaky_relu_

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        out = self.layers[-1](x)
        return out


class CustomDataset(Dataset):
    def __init__(self, set_name):

        result = TH.read_sample_result(set_name, sample_id=0)
        ad = np.array(result[C.key_sample_result_ad])
        sd = np.array(result[C.key_sample_result_sd])
        ai = np.array(result[C.key_sample_result_ai])
        mf = np.array(result[C.key_sample_result_mf])
        r = np.array(result[C.key_sample_result_r])
        t = np.array(result[C.key_sample_result_t])
        re = np.array(result[C.key_sample_result_re])
        te = np.array(result[C.key_sample_result_te])

        ad, sd, ai, mf, r, t = src.leaf_model.training_data.prune_training_data(ad, sd, ai, mf, r, t, re, te)

        self.X = np.column_stack((r,t))
        self.Y = np.column_stack((ad, sd, ai, mf))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train(show_plot=False, layer_count=9, layer_width=10, epochs=300, batch_size=2, learning_rate=0.001, patience=30,
          split=0.1, set_name='training_data'):

    whole_data = CustomDataset(set_name=set_name)
    test_n = int(len(whole_data) * split)
    train_set, test_set = random_split(whole_data, [len(whole_data) - test_n, test_n])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    net = Net(layer_count=layer_count, layer_width=layer_width)

    logging.info(f"Learning rate {learning_rate}")
    logging.info(f"Batch size {batch_size}")
    logging.info(f"Max epochs {epochs}")
    logging.info(f"Test split {split}")
    logging.info(f"Net has {net.layer_count} hidden layers that are {net.layer_width} wide")

    save_name = get_nn_save_name(layer_count=net.layer_count, layer_width=net.layer_width, batch_size=batch_size,
                                 lr=learning_rate, split=split)

    net = net.double()
    logging.info(net)
    # params = net.parameters()

    # critcerion = nn.L1Loss(reduction = 'sum')
    criterion = nn.MSELoss()

    # create your optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []

    n_epochs = epochs
    best_loss = 1e10
    best_epoch_idx = None
    patience_trigger = 0

    for epoch in range(n_epochs):

        net.train(True)
        # batch
        for x, y in train_loader:
            optimizer.zero_grad()
            z = net(x.double())
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())

        net.train(False)
        correct = 0

        # perform a prediction on the test  data
        for x_test, y_test in test_loader:
            z = net(x_test.double())
            test_loss = criterion(z, y_test)

        test_losses.append(test_loss.item())

        logging.info(f"Epoch {epoch}: losses {loss.item():.6f} - {test_loss.item():.6f} (train - test)")

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch_idx = epoch
            patience_trigger = 0
            nn_path = save_name + '.pt'
            save(net, PH.join(PH.path_directory_surface_model(), nn_path))
            logging.info(f"Saved model with test loss {best_loss:.6f} epoch {epoch}")
        else:
            patience_trigger += 1

        if patience_trigger >= patience:
            logging.info(f"Early stopping criteria met: no improvement in test loss in {patience} epochs.")
            break

    print(train_losses)
    logging.info(f"Neural network training finished. Final loss {best_loss}")
    plotter.plot_nn_train_history(train_loss=train_losses, test_loss=test_losses, best_epoch_idx=best_epoch_idx,
                                  dont_show=not show_plot, save_thumbnail=True, file_name=save_name)
    return best_loss


def get_nn_save_name(layer_count, layer_width, batch_size, lr, split):
    # TODO move to FileNames
    name = f"lc{layer_count}_lw{layer_width}_b{batch_size}_lr{lr}_split{split}"
    return name


def predict(r_m, t_m, nn_name='nn_default'):
    net = _load_model(nn_name=nn_name)
    r_m = np.array(r_m)
    t_m = np.array(t_m)
    res = net(from_numpy(np.column_stack([r_m, t_m])))
    res_item = res.detach().numpy()
    ad = np.clip(res_item[:,0], 0., 1.)
    sd = np.clip(res_item[:,1], 0., 1.)
    ai = np.clip(res_item[:,2], 0., 1.)
    mf = np.clip(res_item[:,3], 0., 1.)
    return ad, sd, ai, mf


def _load_model(nn_name):
    try:
        net = load(_get_model_path(nn_name))
        net.eval()
    except ModuleNotFoundError as e:
        logging.error(f"Pytorch could not load requested neural network. "
                      f"This happens if class or file names associated with "
                      f"NN are changed. You must train a new model to fix this.")
        raise
    return net


def _get_model_path(nn_name='nn_default'):
    if not nn_name.endswith('.pt'):
        nn_name = nn_name + '.pt'
    model_path = PH.join(PH.path_directory_surface_model(), nn_name)
    if os.path.exists(model_path):
        return model_path
    else:
        raise FileNotFoundError(f"Model '{model_path}' was not found. Check spelling.")

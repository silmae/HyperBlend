"""
Leaf model implementation as a neural network.

Greatly accelerates prediction time compared to the original
optimization method with some loss to accuracy.

"""

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

import src.leaf_model.training_data as TD
from src.data import path_handling as PH, toml_handling as TH, file_names as FN
from src import constants as C
from src import plotter

# Set manual seed when doing hyperparameter search for comparable results
# between training runs.
torch.manual_seed(666)


class Leafnet(nn.Module):
    """Neural network implementation."""

    def __init__(self, layer_count, layer_width):
        """Initialize neural network with given architecture.

        Number and width of hidden layers as parameters. Activation for all hidden layers is
        leaky relu.
        """

        super(Leafnet, self).__init__()
        input_dim = 2
        output_dim = 4
        self.layer_count = layer_count
        self.layer_width = layer_width
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for i in range(layer_count+1): # +1 because the input layer is also added in the loop
            layer = nn.Linear(current_dim, layer_width)
            self.layers.append(layer)
            current_dim = layer_width
        self.layers.append(nn.Linear(current_dim, output_dim)) # add output layer separately
        self.activation = F.leaky_relu_

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        out = self.layers[-1](x)
        return out


class TrainingData(Dataset):
    """Handles catering the training data from disk to NN."""

    def __init__(self, set_name):
        """Initialize data.

        Prunes badly fitted data points from the set.

        :param set_name:
            Set name from where the data is loaded.
        """

        ad, sd, ai, mf, r, t, re, te = TD.get_training_data(set_name=set_name)
        ad, sd, ai, mf, r, t = TD.prune_training_data(ad, sd, ai, mf, r, t, re, te)

        self.X = np.column_stack((r,t))
        self.Y = np.column_stack((ad, sd, ai, mf))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train(show_plot=False, layer_count=9, layer_width=10, epochs=300, batch_size=2, learning_rate=0.001, patience=30,
          split=0.1, set_name='training_data'):
    """Train the neural network with given parameters.

    Saves the best performing model onto disk with generated name (according to NN architecture and some training
    parameters). You can manually change the name later to
    'nn_default.pt' if you want to replace the old default network that comes from the Git repository.

    :param show_plot:
        Show training history at the end of training. Set False when doing multiple runs.
        The plot is always saved to disk, even if not shown. Default is False.
    :param layer_count:
        Number of hidden layers.
    :param layer_width:
        Width of hidden layers.
    :param epochs:
        Maximum epochs.
    :param batch_size:
        Batch size. Best results with small batches (2). Bigger batches (e.g. 32) train faster
        but reduce accuracy.
    :param learning_rate:
        Learning rate for Adam optimizer. Default 0.001 usually performs best.
    :param patience:
        Early stop training if test loss has not improved in this many epochs.
    :param split:
        Percentage [0,1] of data reserved for testing between epochs. Value between 0.1 and 0.2
        is usually sufficient.
    :param set_name:
        Set name of the training data. Default is OK if you didn't generate training data with a custom name.
    :return:
        Returns the best loss for hyperparameter tuning loops.
    """

    whole_data = TrainingData(set_name=set_name)
    test_n = int(len(whole_data) * split)
    train_set, test_set = random_split(whole_data, [len(whole_data) - test_n, test_n])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    net = Leafnet(layer_count=layer_count, layer_width=layer_width)

    logging.info(f"Learning rate {learning_rate}")
    logging.info(f"Batch size {batch_size}")
    logging.info(f"Max epochs {epochs}")
    logging.info(f"Test split {split}")
    logging.info(f"Net has {net.layer_count} hidden layers that are {net.layer_width} wide")

    save_name = FN.get_nn_save_name(layer_count=net.layer_count, layer_width=net.layer_width, batch_size=batch_size,
                                 lr=learning_rate, split=split)

    net = net.double()
    logging.info(net)

    criterion = nn.MSELoss()

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
            logging.info(f"Saved model with test loss {best_loss:.8f} epoch {epoch}")
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


def predict(r_m, t_m, nn_name='nn_default'):
    """Use neural network to predict HyperBlend leaf model parameters from measured reflectance and transmittance.

    :param r_m:
        Measured reflectance.
    :param t_m:
        Measured transmittance.
    :param nn_name:
        Neural network name. Default name 'nn_default' is used if not given.
        Provide only if you want to use your trained custom NN.
    :return:
        Lists ad, sd, ai, mf (absorption density, scattering desnity, scattering anisotropy, and mixing factor).
        Use ``leaf_commons._convert_raw_params_to_renderable()`` before passing them to rendering method.
    """

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
    """Loads the NN from disk.

    :param nn_name:
        Name of the model file.
    :return:
        Returns loaded NN.
    :exception:
        ModuleNotFoundError can happen if the network was trained when the name
        of this script was something else than what it is now. Your only help
        is to train again.
    """
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
    """Returns path to the NN model.

    :param nn_name:
        Name of the NN.
    :return:
        Returns path to the NN model.
    :exception:
        FileNotFoundError if the model cannot be found.
    """

    if not nn_name.endswith('.pt'):
        nn_name = nn_name + '.pt'
    model_path = PH.join(PH.path_directory_surface_model(), nn_name)
    if os.path.exists(model_path):
        return model_path
    else:
        raise FileNotFoundError(f"Model '{model_path}' was not found. Check spelling.")


def exists():
    """Checks whether the default NN model exists.

    :return:
        True if found, False otherwise.
    """

    nn_name = 'nn_default.pt'
    model_path = PH.join(PH.path_directory_surface_model(), nn_name)
    return os.path.exists(model_path)

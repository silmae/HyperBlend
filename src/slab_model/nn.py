"""
Slab model implementation as a neural network.

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
from torch import from_numpy
import torch.optim as optim

import src.slab_model.training_utils
from src.data import path_handling as PH, file_names as FN
from src import plotter
from src.data.path_handling import path_nn_model


# Set manual seed when doing hyperparameter search for comparable results
# between training runs.
# torch.manual_seed(666)


class Slabnet(nn.Module):
    """Neural network implementation."""

    def __init__(self, layer_count=5, layer_width=1000):
        """Initialize neural network with given architecture.

        Number and width of hidden layers as parameters. Activation for all hidden layers is
        leaky relu.
        """

        super(Slabnet, self).__init__()
        input_dim = 2
        output_dim = 4
        self.layer_count = layer_count
        self.layer_width = layer_width
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for i in range(
            layer_count + 1
        ):  # +1 because the input layer is also added in the loop
            layer = nn.Linear(current_dim, layer_width)
            self.layers.append(layer)
            current_dim = layer_width
        self.layers.append(
            nn.Linear(current_dim, output_dim)
        )  # add output layer separately
        self.activation = F.leaky_relu_

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        out = self.layers[-1](x)
        return out


class TrainingData(Dataset):
    """Handles catering the training data from disk to NN."""

    def __init__(self, training_sim_name: str):
        """Initialize the training data for NN training.

        Badly fitted data points are pruned from the data set.

        :param training_sim_name:
            Name of the training data slab simulation.
        """

        ad, sd, ai, mf, r, t, re, te = src.slab_model.training_utils.get_training_data(
            training_sim_name=training_sim_name
        )
        ad, sd, ai, mf, r, t = src.slab_model.training_utils.prune_training_data(
            ad, sd, ai, mf, r, t, re, te
        )

        self.X = np.column_stack((r, t))
        self.Y = np.column_stack((ad, sd, ai, mf))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train(
    show_plot=False,
    layer_count=10,
    layer_width=1000,
    epochs=300,
    batch_size=2,
    learning_rate=0.001,
    patience=30,
    split=0.1,
    training_sim_name="training_data",
    solver_name=None,
):
    """Train the neural network with given parameters.

    Saves the best performing model onto disk with generated name (according to NN
    architecture and some training parameters). You can manually change the name later to
    'nn_default.pt' if you want to replace the old default network that comes from the
    Git repository.

    :param show_plot: Show training history at the end of training. Set False when
        doing multiple runs. The plot is always saved to disk, even if not shown.
        Default is False.
    :param layer_count: Number of hidden layers.
    :param layer_width: Width of hidden layers.
    :param epochs: Maximum epochs.
    :param batch_size: Batch size. Best results with small batches (2). Bigger
        batches (e.g. 32) train faster but reduce accuracy.
    :param learning_rate: Learning rate for Adam optimizer. Default 0.001 usually performs best.
    :param patience: Early stop training if test loss has not improved in this many epochs.
    :param split: Percentage [0,1] of data reserved for testing between epochs. Value
        between 0.1 and 0.2 is usually sufficient.
    :param training_sim_name: Name of the training data slab simulation. A new solver
        will be saved with this name. Note that the training data actually is another slab
        simulation; just a special kind where we generate the training data points and
        solved their material parameters with the optimization method. No need to change
        the default name unless you generated the data with custom name.
    :param solver_name: Trained solver is saved with this name.
    :return: Returns the best loss for hyperparameter tuning loops.
    """

    whole_data = TrainingData(training_sim_name=training_sim_name)
    test_n = int(len(whole_data) * split)
    train_set, test_set = random_split(whole_data, [len(whole_data) - test_n, test_n])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    net = Slabnet(layer_count=layer_count, layer_width=layer_width)
    best_model_state = net.state_dict()

    logging.info(f"Learning rate {learning_rate}")
    logging.info(f"Batch size {batch_size}")
    logging.info(f"Max epochs {epochs}")
    logging.info(f"Test split {split}")
    logging.info(
        f"Net has {net.layer_count} hidden layers that are {net.layer_width} wide"
    )

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

    # nn_filename = FN.get_nn_save_name(layer_count=layer_count, layer_width=layer_width, batch_size=batch_size,
    #                                   lr=learning_rate, split=split, training_set=training_sim_name)

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

        logging.info(
            f"Epoch {epoch}: losses {loss.item():.6f} - {test_loss.item():.6f} (train - test)"
        )

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch_idx = epoch
            patience_trigger = 0
            best_model_state = net.state_dict()

            save_path = PH.path_nn_model(slab_model_name=solver_name)
            torch.save(best_model_state, save_path)

            logging.info(f"Saved model with test loss {best_loss:.8f} epoch {epoch}")
        else:
            patience_trigger += 1

        if patience_trigger >= patience:
            logging.info(
                f"Early stopping criteria met: no improvement in test loss in {patience} epochs."
            )
            break

    logging.info(train_losses)
    logging.info(f"Neural network training finished. Final loss {best_loss}")
    plotter.plot_nn_train_history(
        train_loss=train_losses,
        test_loss=test_losses,
        best_epoch_idx=best_epoch_idx,
        dont_show=not show_plot,
        save_thumbnail=True,
        solver_name=solver_name,
    )
    return best_loss


def predict(target_refl, target_tran, solver_dirname: str = None):
    """Use neural network to predict HyperBlend slab model parameters from target
    reflectance and transmittance.

    :param target_refl: Target reflectance.
    :param target_tran: Target transmittance.
    :param solver_dirname: Name of the (directory of the) neural network to be used. If
        None, the default is used.
    :return: Lists ad, sd, ai, mf (absorption density, scattering density, scattering
        anisotropy, and mixing factor). Use ``slab_commons._convert_raw_params_to_renderable()``
        before passing them to rendering method.
    """

    if not exists(solver_mame=solver_dirname):
        raise FileNotFoundError(
            f"Neural network with name '{solver_dirname}' not found."
        )

    net = _load_model(solver_dirname=solver_dirname)
    target_refl = np.array(target_refl)
    target_tran = np.array(target_tran)
    res = net(from_numpy(np.column_stack([target_refl, target_tran])))
    res_item = res.detach().numpy()
    ad = np.clip(res_item[:, 0], 0.0, 1.0)
    sd = np.clip(res_item[:, 1], 0.0, 1.0)
    ai = np.clip(res_item[:, 2], 0.0, 1.0)
    mf = np.clip(res_item[:, 3], 0.0, 1.0)
    return ad, sd, ai, mf


def _load_model(solver_dirname: str):
    """Loads the NN from disk.

    :param solver_dirname: Name of the (directory of the) neural network to be used.
        If None, the default is used.
    :return: Returns loaded NN.
    :raises ModuleNotFoundError: If PyTorch cannot load the requested neural network.
    """

    try:
        p = path_nn_model(slab_model_name=solver_dirname)
        net = Slabnet()
        net.load_state_dict(torch.load(p))
        net.double()

        net.eval()
        logging.info(f"NN model loaded from '{p}'")
    except ModuleNotFoundError as e:
        logging.error(f"Pytorch could not load requested neural network.")
        raise
    return net


def exists(solver_mame: str = None):
    """Checks whether NN with given name exists.

    :return:
        True if found, False otherwise.
    """

    return os.path.exists(PH.path_nn_model(slab_model_name=solver_mame))

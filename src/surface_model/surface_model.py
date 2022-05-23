"""
Surface model stuff

"""

import logging
import math
import numpy as np
from scipy.optimize import curve_fit
import torch
import matplotlib.pyplot as plt

from src.data import toml_handling as TH
from src import constants as C
from src.optimization import Optimization
from src.surface_model import fitting_function as FF
from src import plotter

set_name = 'surface_train'


def train(do_points=True, num_points=50):
    """Train surface model."""
    if do_points:
        generate_train_data(num_points)
        o = Optimization(set_name)
        o.run_optimization(prediction_method='optimization')
    fit(show_plot=False, save_params=True)


def generate_train_data(num_points=10):
    """Generate reflectance-transmittance pairs for surface fitting.

    Generated data is saved to disk to be used in optimization.
    """

    data = []
    fake_wl = 1 # Set dummy wavelengths so that the rest of the code is ok with the files
    for i,r in enumerate(np.linspace(0, 0.6, num_points, endpoint=True)):
        for j,t in enumerate(np.linspace(0, 0.6, num_points, endpoint=True)):
            # Do not allow r+t to exceed 1 as it would break conservation of energy
            if r + t >= 0.999999:
                continue
            # ensure some amount of symmetry
            if math.fabs(r-t) > 0.1:
                continue

            wlrt = [fake_wl, r, t]
            data.append(wlrt)
            fake_wl += 1
    logging.info(f"Generated {len(data)} evenly spaced reflectance transmittance pairs.")
    TH.write_target(set_name, data, sample_id=0)


def fit(show_plot=False, save_params=False, use_nn=True):

    if use_nn:
        fit_nn(show_plot=show_plot, save_params=save_params)
    else:
        fit_surface(show_plot=show_plot, save_params=save_params)


def fit_surface(show_plot=False, save_params=False):
    """Fit surfaces.

    Surface fitting parameters written to disk and plots shown if show_plot=True.
    :param save_params:
    """
    # ids = FH.list_finished_sample_ids(set_name)
    # for _, sample_id in enumerate(ids):
    result = TH.read_sample_result(set_name, sample_id=0)
    ad = np.array(result[C.key_sample_result_ad])
    sd = np.array(result[C.key_sample_result_sd])
    ai = np.array(result[C.key_sample_result_ai])
    mf = np.array(result[C.key_sample_result_mf])
    r  = np.array(result[C.key_sample_result_r])
    t  = np.array(result[C.key_sample_result_t])
    re = np.array(result[C.key_sample_result_re])
    te = np.array(result[C.key_sample_result_te])

    max_error = 0.01
    low_cut = 0.0
    bad = [(a > max_error or b > max_error) for a,b in zip(re, te)]
    # bad = np.where(bad)[0]
    low_cut = [(a < low_cut or b < low_cut) for a,b in zip(r, t)]
    to_delete = np.logical_or(bad, low_cut)
    # to_delete = bad

    to_delete = np.where(to_delete)[0]
    ad = np.delete(ad, to_delete)
    sd = np.delete(sd, to_delete)
    ai = np.delete(ai, to_delete)
    mf = np.delete(mf, to_delete)
    r  = np.delete(r , to_delete)
    t  = np.delete(t , to_delete)

    variable_lol = [ad, sd, ai, mf]
    variable_names = ['ad', 'sd', 'ai', 'mf']

    result_dict = {}

    for i, variable in enumerate(variable_lol):
        # get fit parameters from scipy curve fit
        # https://stackoverflow.com/questions/56439930/how-to-use-the-datasets-to-fit-the-3d-surface

        zlabel = variable_names[i]
        failed = False

        try:
            fittable = FF.function_exp
            if variable_names[i] == 'ai':
                fittable = FF.function_polynomial
            if variable_names[i] == 'sd':
                fittable = FF.function_log
            if variable_names[i] == 'mf':
                fittable = FF.function_exp

            parameters, _ = curve_fit(fittable, [r, t], variable, p0=FF.get_x0())
            result_dict[variable_names[i]] = parameters
            if show_plot:
                plotter.plot_3d_rt(r,t,variable,zlabel,z_intensity=None,surface_parameters=parameters,fittable=fittable)
        except RuntimeError as re:
            logging.error(f'Failed to fit for parameter {variable_names[i]}')
            if show_plot:
                plotter.plot_3d_rt(r, t, variable, zlabel)
                failed = True

    if not failed:
        if save_params:
            print(f'Saving surface model parameters')
            TH.write_surface_model_parameters(result_dict)
    else:
        raise RuntimeError(f'Failed to fit all parameters. The result will not be saved.')


###########################
# From tutorial: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
###########################

import torch
import torch.nn as nn
import torch.nn.functional as F
# datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # 5*5 from image dimension
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class CustomDataset(Dataset):
    def __init__(self):

        result = TH.read_sample_result(set_name, sample_id=0)
        ad = np.array(result[C.key_sample_result_ad])
        sd = np.array(result[C.key_sample_result_sd])
        ai = np.array(result[C.key_sample_result_ai])
        mf = np.array(result[C.key_sample_result_mf])
        r = np.array(result[C.key_sample_result_r])
        t = np.array(result[C.key_sample_result_t])
        re = np.array(result[C.key_sample_result_re])
        te = np.array(result[C.key_sample_result_te])

        max_error = 0.01
        low_cut = 0.0
        bad = [(a > max_error or b > max_error) for a, b in zip(re, te)]
        # bad = np.where(bad)[0]
        low_cut = [(a < low_cut or b < low_cut) for a, b in zip(r, t)]
        to_delete = np.logical_or(bad, low_cut)
        # to_delete = bad

        to_delete = np.where(to_delete)[0]
        ad = np.delete(ad, to_delete)
        sd = np.delete(sd, to_delete)
        ai = np.delete(ai, to_delete)
        mf = np.delete(mf, to_delete)
        r = np.delete(r, to_delete)
        t = np.delete(t, to_delete)

        self.X = np.column_stack((r,t))
        self.Y = np.column_stack((ad, sd, mf, ai))

        variable_lol = [ad, sd, ai, mf]
        variable_names = ['ad', 'sd', 'ai', 'mf']

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def fit_nn(show_plot=False, save_params=False):
    """Fit surfaces.

    Surface fitting parameters written to disk and plots shown if show_plot=True.
    :param save_params:
    """

    # ids = FH.list_finished_sample_ids(set_name)
    # for _, sample_id in enumerate(ids):

#     something something



    whole_data = CustomDataset()
    test_n = 100
    train_set, test_set = torch.utils.data.random_split(whole_data, [len(whole_data)-test_n, test_n])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)


    net = Net()
    net = net.double()
    print(net)

    criterion = nn.MSELoss()


    # for x,y in train_loader:
    #     output = net(x.float())
    #     loss = criterion(output, y)
    #     print(loss)


    # create your optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    train_losses = []
    test_losses = []
    accuracy_list = []

    n_epochs = 50

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
        # perform a prediction on the validation  data
        for x_test, y_test in test_loader:
            z = net(x_test.double())
            test_loss = criterion(z, y_test)

        test_losses.append(test_loss.item())

            # _, yhat = torch.max(z.data, 1)
            # correct += (yhat == y_test).sum().item()
        # accuracy = correct / test_n
        # accuracy_list.append(accuracy)

    import matplotlib.pyplot as plt
    plt.plot(train_losses, label="Train loss")
    plt.plot(test_losses, label="Test loss")
    plt.legend()
    plt.show()

    print(train_losses)

    # train_model(n_epochs)

    # quit(0)

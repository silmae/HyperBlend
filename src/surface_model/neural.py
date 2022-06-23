
##########################
# From tutorial: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
##########################

import logging
import numpy as np
# import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import save
from torch import load
from torch import from_numpy
import torch.optim as optim

from src.data import path_handling as PH
from src.data import toml_handling as TH
from src import constants as C
from src import plotter


set_name = 'surface_train' # use same set name as surface_model

model_path = PH.join(PH.path_directory_surface_model(), "nn_default.pt")

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # 5*5 from image dimension
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 512)

        self.fc31 = nn.Linear(512, 2048)
        self.fc32 = nn.Linear(2048, 512)

        self.fc4 = nn.Linear(512, 64)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(16, 4)

        # nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.xavier_normal_(self.fc3.weight)
        # nn.init.xavier_normal_(self.fc4.weight)
        # nn.init.xavier_normal_(self.fc5.weight)
        # nn.init.xavier_normal_(self.fc6.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = F.relu(self.fc31(x))
        x = F.relu(self.fc32(x))

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
        self.Y = np.column_stack((ad, sd, ai, mf))

        variable_lol = [ad, sd, ai, mf]
        variable_names = ['ad', 'sd', 'ai', 'mf']

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def fit_nn(show_plot=False, save_params=False, epochs=150):
    """Fit surfaces.

    Surface fitting parameters written to disk and plots shown if show_plot=True.
    :param save_params:
    """

    whole_data = CustomDataset()
    test_n = int(len(whole_data) * 0.2)
    batch_size = 16
    train_set, test_set = random_split(whole_data, [len(whole_data)-test_n, test_n])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    net = Net()
    net = net.double()
    print(net)

    criterion = nn.MSELoss()

    # create your optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_losses = []
    test_losses = []

    n_epochs = epochs
    best_loss = 1e10
    best_epoch_idx = None
    patience = 50
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
            save(net, model_path)
            logging.info(f"Saved model with test loss {best_loss:.6f} epoch {epoch}")
        else:
            patience_trigger += 1

        if patience_trigger >= patience:
            logging.info(f"Early stopping criteria met: no improvement in test loss in {patience} epochs.")
            break

            # _, yhat = torch.max(z.data, 1)
            # correct += (yhat == y_test).sum().item()
        # accuracy = correct / test_n
        # accuracy_list.append(accuracy)

    # import matplotlib.pyplot as plt
    # plt.plot(train_losses, label="Train loss")
    # plt.plot(test_losses, label="Test loss")
    # plt.legend()
    # plt.show()
    plotter.plot_nn_train_history(train_loss=train_losses, test_loss=test_losses, best_epoch_idx=best_epoch_idx, save_thumbnail=True, dont_show=not show_plot)

    print(train_losses)


def predict_nn(r,t):
    net = load_model()
    r = np.array(r)
    t = np.array(t)
    res = net(from_numpy(np.column_stack([r,t])))
    res_item = res.detach().numpy()
    ad = np.clip(res_item[:,0], 0., 1.)
    sd = np.clip(res_item[:,1], 0., 1.)
    ai = np.clip(res_item[:,2], 0., 1.)
    mf = np.clip(res_item[:,3], 0., 1.)
    return ad, sd, ai, mf


def load_model():
    net = load(model_path)
    net.eval()
    return net

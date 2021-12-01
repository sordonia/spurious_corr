import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import RandomSampler


def make_toy_dataset(
    n,
    d_noise,
    p_correlation,
    mean_causal,
    var_causal,
    mean_spurious,
    var_spurious,
    noise_type="gaussian",
    mean_noise=0,
    var_noise=1,
    train=True,
):
    """
    explicit memorization setting from https://arxiv.org/pdf/2005.04345.pdf
    """
    groups = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    n_groups = len(groups)

    y_list, a_list, x_causal_list, x_spurious_list, g_list = [], [], [], [], []
    for group_idx, (y_value, a_value) in enumerate(groups):
        if train:
            if y_value == a_value:
                n_group = int(np.round(n / 2 * p_correlation))
            else:
                n_group = int(np.round(n / 2 * (1 - p_correlation)))
        else:
            n_group = int(n / n_groups)

        y_list.append(np.ones(n_group) * y_value)
        a_list.append(np.ones(n_group) * a_value)
        g_list.append(np.ones(n_group) * group_idx)
        x_causal_list.append(
            np.random.normal(y_value * mean_causal, var_causal ** 0.5, n_group).reshape(
                n_group, 1
            )
        )
        x_spurious_list.append(
            np.random.normal(
                a_value * mean_spurious, var_spurious ** 0.5, n_group
            ).reshape(n_group, 1)
        )

    x_noise = np.random.multivariate_normal(
        mean=mean_noise * np.ones(d_noise),
        cov=np.eye(d_noise) * var_noise / d_noise,
        size=n,
    )
    y = np.concatenate(y_list)
    a = np.concatenate(a_list)
    g = np.concatenate(g_list)
    x_causal = np.vstack(x_causal_list)
    x_spurious = np.vstack(x_spurious_list)
    x = np.hstack([x_causal, x_spurious, x_noise])
    return x, y, g, n_groups


def generate_dataset(data_generation_fn, data_args):
    train_x, train_y, train_g, n_groups = data_generation_fn(**data_args, train=True)
    test_x, test_y, test_g, _ = data_generation_fn(**data_args, train=False)
    full_data = (train_x, train_y, train_g), (test_x, test_y, test_g)
    return full_data, n_groups


def data_to_torch(X, y):
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    return X, y


class LiangDataset(LightningDataModule):
    def __init__(self, args):
        self.data_args = {
            "n": 3000,
            "d_noise": 100,
            "p_correlation": 0.995,
            "mean_causal": 0.1,
            "var_causal": 0.001,
            "mean_spurious": 5,
            "var_spurious": 1,
            "mean_noise": 0,
            "var_noise": 1,
        }

        train, test = generate_dataset(make_toy_dataset, self.data_args)[0]
        self.X_spu, self.y_spu, self.g_spu = train
        self.X_test, self.y_test, self.g_test = test
        self.y_spu[self.y_spu == -1] = 0
        self.y_test[self.y_test == -1] = 0

        if args.batch_size == -1:
            # full batch
            args.batch_size = len(self.X_spu)

        self.args = args
        self.batch_size = args.batch_size
        self.train_set = torch.utils.data.TensorDataset(
            torch.from_numpy(self.X_spu).float(),
            torch.from_numpy(self.y_spu).float(),
            torch.from_numpy(self.g_spu).long(),
        )
        self.test_set = torch.utils.data.TensorDataset(
            torch.from_numpy(self.X_test).float(),
            torch.from_numpy(self.y_test).float(),
            torch.from_numpy(self.g_test).long(),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=32, shuffle=False)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True
        )

    def plot_model(self, model, output_file):
        from matplotlib import pyplot as plt

        X, y = self.X_spu, self.y_spu
        device = next(model.parameters()).device

        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole gid
        data_args = self.data_args
        if data_args:
            x_noise = np.random.multivariate_normal(
                mean=data_args["mean_noise"] * np.ones(data_args["d_noise"]),
                cov=np.eye(data_args["d_noise"])
                * data_args["var_noise"]
                / data_args["d_noise"],
                size=xx.ravel().shape[0],
            )
            stack = np.c_[xx.ravel(), yy.ravel()]
            x = np.hstack((stack, x_noise))
            Z = (
                model.predict(torch.from_numpy(x).to(device).float())
                .detach()
                .cpu()
                .numpy()
            )
        else:
            assert False

        Z = Z.reshape(xx.shape)

        # Plot the contour and training examples
        plt.figure(figsize=(4, 4), dpi=80, facecolor="w", edgecolor="k")
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
        plt.savefig(output_file)

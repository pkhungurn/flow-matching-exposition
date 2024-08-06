import math
import os
from typing import Callable

import torch
from matplotlib import pyplot, ticker
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical, Distribution

from data._20240806.twodim_datasets.constants import DATA_20240806_TWODIM_DATASETS_PREFIX
from pytasuku import Workspace
from shion.core.load_save import torch_save, torch_load


def get_dist_00():
    means = []
    for i in range(5):
        theta = 2 * math.pi * i / 5
        means.append([2 * math.cos(theta), 2 * math.sin(theta)])
    covariance_matrices = [
        [[0.25, 0.0], [0.0, 0.25]]
        for i in range(5)
    ]
    gaussians = MultivariateNormal(loc=torch.tensor(means), covariance_matrix=torch.tensor(covariance_matrices))

    dist = MixtureSameFamily(
        mixture_distribution=Categorical(probs=torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])),
        component_distribution=gaussians)

    return dist


class TwoDimDatasetArgs:
    def __init__(self,
                 prefix: str,
                 dist_func: Callable[[], Distribution],
                 axis_x_lim=(-3, 3),
                 axis_y_lim=(-3, 3),
                 mesh_grid_size: int = 800,
                 num_samples: int = 1_000_000):
        self.prefix = prefix
        self.num_samples = num_samples
        self.mesh_grid_size = mesh_grid_size
        self.axis_y_lim = axis_y_lim
        self.axis_x_lim = axis_x_lim
        self.dist_func = dist_func

    def heatmap_file_name(self):
        return f"{self.prefix}/heatmap.png"

    def get_interval_samples(self, x_min: float, x_max: float, num_steps: int):
        width = x_max - x_min
        dx = width / num_steps
        start = x_min + dx / 2
        end = x_max - dx / 2
        return torch.linspace(start, end, num_steps)

    def draw_grid(self, axis):
        axis.set_xlim(self.axis_x_lim[0], self.axis_x_lim[1])
        axis.set_ylim(self.axis_y_lim[0], self.axis_y_lim[1])

        axis.xaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))
        axis.yaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))

        axis.tick_params(axis='both', which='minor', length=0)  # remove minor tick lines

        axis.grid()

    def create_heatmap(self):
        dist = self.dist_func()
        x = self.get_interval_samples(self.axis_x_lim[0], self.axis_x_lim[1], self.mesh_grid_size)
        y = self.get_interval_samples(self.axis_y_lim[0], self.axis_y_lim[1], self.mesh_grid_size)
        X, Y = torch.meshgrid(x, y)
        XY = torch.cat([Y.unsqueeze(2), X.unsqueeze(2)], dim=2)
        prob = torch.exp(dist.log_prob(XY))
        p = prob.numpy()

        fig, ((axis)) = pyplot.subplots(1, 1, figsize=(6, 6))
        self.draw_grid(axis)
        axis.imshow(
            p,
            interpolation='antialiased',
            origin='lower',
            extent=[self.axis_x_lim[0], self.axis_x_lim[1], self.axis_y_lim[0], self.axis_y_lim[1]],
            vmin=0,
            vmax=0.2 / (2 * math.pi * 0.5),
            clip_on=True)

        os.makedirs(self.prefix, exist_ok=True)
        pyplot.savefig(self.heatmap_file_name())
        pyplot.close()

    def dataset_file_name(self):
        return f"{self.prefix}/dataset.pt"

    def scatter_plot_file_name(self):
        return f"{self.prefix}/scatter_plot.png"

    def create_scatter_plot(self):
        data = torch_load(self.dataset_file_name())
        x = data[0:10000, 0].numpy()
        y = data[0:10000, 1].numpy()
        fig, ((axis)) = pyplot.subplots(1, 1, figsize=(6, 6))
        self.draw_grid(axis)
        pyplot.scatter(x, y, alpha=0.01)
        os.makedirs(self.prefix, exist_ok=True)
        pyplot.savefig(self.scatter_plot_file_name())
        pyplot.close()

    def create_dataset(self):
        dist = self.dist_func()
        samples = dist.sample(torch.Size([self.num_samples]))
        os.makedirs(self.prefix, exist_ok=True)
        torch_save(samples, self.dataset_file_name())

    def all_command_name(self):
        return f"{self.prefix}/all"

    def define_tasks(self, workspace: Workspace):
        all_tasks = []

        workspace.create_file_task(self.heatmap_file_name(), [], self.create_heatmap)
        all_tasks.append(self.heatmap_file_name())

        workspace.create_file_task(self.dataset_file_name(), [], self.create_dataset)
        all_tasks.append(self.dataset_file_name())

        workspace.create_file_task(self.scatter_plot_file_name(), [self.dataset_file_name()], self.create_scatter_plot)
        all_tasks.append(self.scatter_plot_file_name())

        workspace.create_command_task(self.all_command_name(), all_tasks)


def define_data_20240806_dataset_00_tasks(workspace: Workspace):
    all_tasks = []

    args = TwoDimDatasetArgs(f"{DATA_20240806_TWODIM_DATASETS_PREFIX}/dataset_00", get_dist_00)
    args.define_tasks(workspace)
    all_tasks.append(args.all_command_name())

    workspace.create_command_task(f"{DATA_20240806_TWODIM_DATASETS_PREFIX}/all", all_tasks)


if __name__ == "__main__":
    dist = get_dist_00()
    args = TwoDimDatasetArgs(f"{DATA_20240806_TWODIM_DATASETS_PREFIX}/dataset_00", get_dist_00)
    args.create_scatter_plot()

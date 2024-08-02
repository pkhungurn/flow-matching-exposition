import os

import numpy
from matplotlib import pyplot, ticker

from data._20240729.probdist.tasks import get_gaussian_probdist
from data._20240802.gaussian_viz.constants import DATA_20240802_GAUSSIAN_VIZ_PREFIX
from pytasuku import Workspace


def plot_gaussian(file_name: str,
                  mu_x: float = 0.0,
                  mu_y: float = 0.0,
                  sigma: float = 1.0,
                  axis_x_lim=(-2, 2),
                  axis_y_lim=(-2, 2),
                  v_min=None,
                  v_max=None,
                  norm=None,
                  draw_circles: bool = True):
    fig, ((axis)) = pyplot.subplots(1, 1, figsize=(6, 6))

    p = get_gaussian_probdist(mu_x, mu_y, sigma)

    axis.imshow(
        p,
        interpolation='antialiased',
        origin='lower',
        extent=[axis_x_lim[0], axis_x_lim[1], axis_y_lim[0], axis_y_lim[1]],
        norm=norm,
        vmin=v_min,
        vmax=v_max,
        clip_on=True)

    axis.set_xlim(axis_x_lim[0], axis_x_lim[1])
    axis.set_ylim(axis_y_lim[0], axis_y_lim[1])

    axis.xaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))
    axis.yaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))

    axis.tick_params(axis='both', which='minor', length=0)  # remove minor tick lines

    axis.grid()

    if draw_circles:
        circle = pyplot.Circle((mu_x,mu_y), sigma, color='r', fill=False)
        axis.add_patch(circle)

        circle = pyplot.Circle((mu_x,mu_y), 0.05, color='r')
        axis.add_patch(circle)

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    pyplot.savefig(file_name)
    pyplot.close(fig)


def define_data_20240802_gaussian_viz_tasks(workspace: Workspace):
    all_tasks = []

    v_min = 0
    v_max = 1.0 / (2.0*numpy.pi*0.5)

    GAUSSIAN_VIZ_00_FILE_NAME = f"{DATA_20240802_GAUSSIAN_VIZ_PREFIX}/gaussian_viz_00.png"
    workspace.create_file_task(
        GAUSSIAN_VIZ_00_FILE_NAME,
        [],
        lambda: plot_gaussian(
            GAUSSIAN_VIZ_00_FILE_NAME,
            mu_x=0.0,
            mu_y=0.0,
            sigma=1.0,
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            norm="linear",
            v_min=v_min,
            v_max=v_max)
    )
    all_tasks.append(GAUSSIAN_VIZ_00_FILE_NAME)

    GAUSSIAN_VIZ_00_NO_CIRCLES_FILE_NAME = f"{DATA_20240802_GAUSSIAN_VIZ_PREFIX}/gaussian_viz_00_no_circles.png"
    workspace.create_file_task(
        GAUSSIAN_VIZ_00_NO_CIRCLES_FILE_NAME,
        [],
        lambda: plot_gaussian(
            GAUSSIAN_VIZ_00_NO_CIRCLES_FILE_NAME,
            mu_x=0.0,
            mu_y=0.0,
            sigma=1.0,
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            norm="linear",
            v_min=v_min,
            v_max=v_max,
            draw_circles=False)
    )
    all_tasks.append(GAUSSIAN_VIZ_00_NO_CIRCLES_FILE_NAME)

    GAUSSIAN_VIZ_01_FILE_NAME = f"{DATA_20240802_GAUSSIAN_VIZ_PREFIX}/gaussian_viz_01.png"
    workspace.create_file_task(
        GAUSSIAN_VIZ_01_FILE_NAME,
        [],
        lambda: plot_gaussian(
            GAUSSIAN_VIZ_01_FILE_NAME,
            mu_x=2.0,
            mu_y=-2.0,
            sigma=0.5,
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            norm="linear",
            v_min=v_min,
            v_max=v_max)
    )
    all_tasks.append(GAUSSIAN_VIZ_01_FILE_NAME)

    GAUSSIAN_VIZ_01_NO_CIRCLES_FILE_NAME = f"{DATA_20240802_GAUSSIAN_VIZ_PREFIX}/gaussian_viz_01_no_circles.png"
    workspace.create_file_task(
        GAUSSIAN_VIZ_01_NO_CIRCLES_FILE_NAME,
        [],
        lambda: plot_gaussian(
            GAUSSIAN_VIZ_01_NO_CIRCLES_FILE_NAME,
            mu_x=2.0,
            mu_y=-2.0,
            sigma=0.5,
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            norm="linear",
            v_min=v_min,
            v_max=v_max,
            draw_circles=False)
    )
    all_tasks.append(GAUSSIAN_VIZ_01_NO_CIRCLES_FILE_NAME)

    workspace.create_command_task(f"{DATA_20240802_GAUSSIAN_VIZ_PREFIX}/all", all_tasks)
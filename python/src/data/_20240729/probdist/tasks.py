import os

import numpy
from matplotlib import pyplot, ticker

from data._20240729.probdist.constants import DATA_20240729_PROBDIST_PREFIX
from data._20240729.translations.tasks import get_hoshihina_image
from pytasuku import Workspace


def get_standard_uniform_probdist():
    delta = 6.0 / 1800
    x = y = numpy.arange(-3.0, 3.0, delta)
    X, Y = numpy.meshgrid(x, y)
    p = (X >= 0).astype(float) * (X <= 1).astype(float) * (Y >= 0).astype(float) * (Y <= 1).astype(float)
    return p


def get_larger_uniform_probdist():
    delta = 6.0 / 1800
    x = y = numpy.arange(-3.0, 3.0, delta)
    X, Y = numpy.meshgrid(x, y)
    p = (X >= -1).astype(float) * (X <= 1).astype(float) * (Y >= -1).astype(float) * (Y <= 1).astype(float)
    return p / 4.0


def get_gaussian_probdist(mu_x: float = 0.0, mu_y: float = 0.0, sigma: float = 1.0):
    delta = 6.0 / 1800
    x = y = numpy.arange(-3.0, 3.0, delta)
    X, Y = numpy.meshgrid(x, y)
    p = numpy.exp(-((X - mu_x) ** 2 + (Y - mu_y) ** 2) / (2*sigma**2)) / (2 * numpy.pi * sigma)
    return p

def get_hoshihina_probdist():
    image = get_hoshihina_image()
    p = image[:,:,0]
    return p * (600*600) / p.sum() / 36


def plot_probdist(file_name: str, p, axis_x_lim=(-2, 2), axis_y_lim=(-2, 2), v_min=None, v_max=None, norm=None):
    fig, ((axis)) = pyplot.subplots(1, 1, figsize=(6, 6))

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

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    pyplot.savefig(file_name)
    pyplot.close(fig)


def define_data_20240729_probdist_tasks(workspace: Workspace):
    all_tasks = []

    v_min = 0
    v_max = 1

    PROBDIST_00_FILE_NAME = f"{DATA_20240729_PROBDIST_PREFIX}/_00.png"
    workspace.create_file_task(
        PROBDIST_00_FILE_NAME,
        [],
        lambda: plot_probdist(
            PROBDIST_00_FILE_NAME,
            get_standard_uniform_probdist(),
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            norm="linear")
    )
    all_tasks.append(PROBDIST_00_FILE_NAME)

    PROBDIST_00_TO_SCALE_FILE_NAME = f"{DATA_20240729_PROBDIST_PREFIX}/_00_to_scale.png"
    workspace.create_file_task(
        PROBDIST_00_TO_SCALE_FILE_NAME,
        [],
        lambda: plot_probdist(
            PROBDIST_00_TO_SCALE_FILE_NAME,
            get_standard_uniform_probdist(),
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            v_min=v_min,
            v_max=v_max)
    )
    all_tasks.append(PROBDIST_00_TO_SCALE_FILE_NAME)

    PROBDIST_01_FILE_NAME = f"{DATA_20240729_PROBDIST_PREFIX}/_01.png"
    workspace.create_file_task(
        PROBDIST_01_FILE_NAME,
        [],
        lambda: plot_probdist(
            PROBDIST_01_FILE_NAME,
            get_larger_uniform_probdist(),
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            norm="linear")
    )
    all_tasks.append(PROBDIST_01_FILE_NAME)

    PROBDIST_01_TO_SCALE_FILE_NAME = f"{DATA_20240729_PROBDIST_PREFIX}/_01_to_scale.png"
    workspace.create_file_task(
        PROBDIST_01_TO_SCALE_FILE_NAME,
        [],
        lambda: plot_probdist(
            PROBDIST_01_TO_SCALE_FILE_NAME,
            get_larger_uniform_probdist(),
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            v_min=v_min,
            v_max=v_max)
    )
    all_tasks.append(PROBDIST_01_TO_SCALE_FILE_NAME)

    PROBDIST_02_FILE_NAME = f"{DATA_20240729_PROBDIST_PREFIX}/_02.png"
    workspace.create_file_task(
        PROBDIST_02_FILE_NAME,
        [],
        lambda: plot_probdist(
            PROBDIST_02_FILE_NAME,
            get_hoshihina_probdist(),
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            norm="linear")
    )
    all_tasks.append(PROBDIST_02_FILE_NAME)

    PROBDIST_02_TO_SCALE_FILE_NAME = f"{DATA_20240729_PROBDIST_PREFIX}/_02_to_scale.png"
    workspace.create_file_task(
        PROBDIST_02_TO_SCALE_FILE_NAME,
        [],
        lambda: plot_probdist(
            PROBDIST_02_TO_SCALE_FILE_NAME,
            get_hoshihina_probdist(),
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            v_min=v_min,
            v_max=v_max)
    )
    all_tasks.append(PROBDIST_02_TO_SCALE_FILE_NAME)


    v_max = 1.0 / (2*numpy.pi*0.5)

    GAUSSIAN_00_FILE_NAME = f"{DATA_20240729_PROBDIST_PREFIX}/gaussian_00.png"
    workspace.create_file_task(
        GAUSSIAN_00_FILE_NAME,
        [],
        lambda: plot_probdist(
            GAUSSIAN_00_FILE_NAME,
            get_gaussian_probdist(0,0,1),
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            norm="linear")
    )
    all_tasks.append(GAUSSIAN_00_FILE_NAME)

    GAUSSIAN_00_TO_SCALE_FILE_NAME = f"{DATA_20240729_PROBDIST_PREFIX}/gaussian_00_to_scale.png"
    workspace.create_file_task(
        GAUSSIAN_00_TO_SCALE_FILE_NAME,
        [],
        lambda: plot_probdist(
            GAUSSIAN_00_TO_SCALE_FILE_NAME,
            get_gaussian_probdist(0,0,1),
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            v_min=v_min,
            v_max=v_max)
    )
    all_tasks.append(GAUSSIAN_00_TO_SCALE_FILE_NAME)

    GAUSSIAN_01_FILE_NAME = f"{DATA_20240729_PROBDIST_PREFIX}/gaussian_01.png"
    workspace.create_file_task(
        GAUSSIAN_01_FILE_NAME,
        [],
        lambda: plot_probdist(
            GAUSSIAN_01_FILE_NAME,
            get_gaussian_probdist(-1,1,0.5),
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            norm="linear")
    )
    all_tasks.append(GAUSSIAN_01_FILE_NAME)

    GAUSSIAN_01_TO_SCALE_FILE_NAME = f"{DATA_20240729_PROBDIST_PREFIX}/gaussian_01_to_scale.png"
    workspace.create_file_task(
        GAUSSIAN_01_TO_SCALE_FILE_NAME,
        [],
        lambda: plot_probdist(
            GAUSSIAN_01_TO_SCALE_FILE_NAME,
            get_gaussian_probdist(-1,1,0.5),
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            v_min=v_min,
            v_max=v_max)
    )
    all_tasks.append(GAUSSIAN_01_TO_SCALE_FILE_NAME)

    GAUSSIAN_02_FILE_NAME = f"{DATA_20240729_PROBDIST_PREFIX}/gaussian_02.png"
    workspace.create_file_task(
        GAUSSIAN_02_FILE_NAME,
        [],
        lambda: plot_probdist(
            GAUSSIAN_02_FILE_NAME,
            get_gaussian_probdist(2,-1,2),
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            norm="linear")
    )
    all_tasks.append(GAUSSIAN_02_FILE_NAME)

    GAUSSIAN_02_TO_SCALE_FILE_NAME = f"{DATA_20240729_PROBDIST_PREFIX}/gaussian_02_to_scale.png"
    workspace.create_file_task(
        GAUSSIAN_02_TO_SCALE_FILE_NAME,
        [],
        lambda: plot_probdist(
            GAUSSIAN_02_TO_SCALE_FILE_NAME,
            get_gaussian_probdist(2,-1,2),
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            v_min=v_min,
            v_max=v_max)
    )
    all_tasks.append(GAUSSIAN_02_TO_SCALE_FILE_NAME)

    GAUSSIAN_03_TO_SCALE_FILE_NAME = f"{DATA_20240729_PROBDIST_PREFIX}/gaussian_03_to_scale.png"
    workspace.create_file_task(
        GAUSSIAN_03_TO_SCALE_FILE_NAME,
        [],
        lambda: plot_probdist(
            GAUSSIAN_03_TO_SCALE_FILE_NAME,
            get_gaussian_probdist(1,-1,1),
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            v_min=v_min,
            v_max=v_max)
    )
    all_tasks.append(GAUSSIAN_03_TO_SCALE_FILE_NAME)

    GAUSSIAN_04_TO_SCALE_FILE_NAME = f"{DATA_20240729_PROBDIST_PREFIX}/gaussian_04_to_scale.png"
    workspace.create_file_task(
        GAUSSIAN_04_TO_SCALE_FILE_NAME,
        [],
        lambda: plot_probdist(
            GAUSSIAN_04_TO_SCALE_FILE_NAME,
            get_gaussian_probdist(0,0,2),
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            v_min=v_min,
            v_max=v_max)
    )
    all_tasks.append(GAUSSIAN_04_TO_SCALE_FILE_NAME)

    GAUSSIAN_05_TO_SCALE_FILE_NAME = f"{DATA_20240729_PROBDIST_PREFIX}/gaussian_05_to_scale.png"
    workspace.create_file_task(
        GAUSSIAN_05_TO_SCALE_FILE_NAME,
        [],
        lambda: plot_probdist(
            GAUSSIAN_05_TO_SCALE_FILE_NAME,
            get_gaussian_probdist(0,0,0.5),
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            v_min=v_min,
            v_max=v_max)
    )
    all_tasks.append(GAUSSIAN_05_TO_SCALE_FILE_NAME)

    GAUSSIAN_06_TO_SCALE_FILE_NAME = f"{DATA_20240729_PROBDIST_PREFIX}/gaussian_06_to_scale.png"
    workspace.create_file_task(
        GAUSSIAN_06_TO_SCALE_FILE_NAME,
        [],
        lambda: plot_probdist(
            GAUSSIAN_06_TO_SCALE_FILE_NAME,
            get_gaussian_probdist(1,-1,0.5),
            axis_x_lim=(-3, 3),
            axis_y_lim=(-3, 3),
            v_min=v_min,
            v_max=v_max)
    )
    all_tasks.append(GAUSSIAN_06_TO_SCALE_FILE_NAME)


    workspace.create_command_task(f"{DATA_20240729_PROBDIST_PREFIX}/all", all_tasks)

import os

from matplotlib import pyplot, transforms, ticker

from data._20240729.constants import HOSHIHINA_600_FILE_NAME
from data._20240729.scalings.constants import DATA_20240729_SCALINGS_PREFIX
from data._20240729.translations.tasks import get_hoshihina_image, plot_image
from pytasuku import Workspace


def plot_scaled_image(
        file_name: str,
        x_scale: float,
        y_scale: float,
        axis_x_lim=(-2, 2), axis_y_lim=(-2, 2)):
    # matplotlib.rc('font', size=18)
    fig, ((axis)) = pyplot.subplots(1, 1, figsize=(6, 6))

    image = get_hoshihina_image()
    xform = transforms.Affine2D().scale(x_scale, y_scale)
    plot_image(axis, image, transform=xform)

    axis.set_xlim(axis_x_lim[0], axis_x_lim[1])
    axis.set_ylim(axis_y_lim[0], axis_y_lim[1])

    axis.xaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))
    axis.yaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))

    axis.tick_params(axis='both', which='minor', length=0)  # remove minor tick lines

    axis.grid()

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    pyplot.savefig(file_name)
    pyplot.close(fig)


def define_data_20240729_scalings_tasks(workspace: Workspace):
    all_tasks = []

    SCALING_00_FILE_NAME = f"{DATA_20240729_SCALINGS_PREFIX}/hoshihina_scaling_00.png"
    workspace.create_file_task(
        SCALING_00_FILE_NAME,
        [HOSHIHINA_600_FILE_NAME],
        lambda: plot_scaled_image(SCALING_00_FILE_NAME, 1.0, 1.0, axis_x_lim=(-3, 3), axis_y_lim=(-3, 3)))
    all_tasks.append(SCALING_00_FILE_NAME)

    SCALING_01_FILE_NAME = f"{DATA_20240729_SCALINGS_PREFIX}/hoshihina_scaling_01.png"
    workspace.create_file_task(
        SCALING_01_FILE_NAME,
        [HOSHIHINA_600_FILE_NAME],
        lambda: plot_scaled_image(SCALING_01_FILE_NAME, 2.0, 2.0, axis_x_lim=(-3, 3), axis_y_lim=(-3, 3)))
    all_tasks.append(SCALING_01_FILE_NAME)

    SCALING_02_FILE_NAME = f"{DATA_20240729_SCALINGS_PREFIX}/hoshihina_scaling_02.png"
    workspace.create_file_task(
        SCALING_02_FILE_NAME,
        [HOSHIHINA_600_FILE_NAME],
        lambda: plot_scaled_image(SCALING_02_FILE_NAME, -1.0, -1.0, axis_x_lim=(-2, 2), axis_y_lim=(-2, 2)))
    all_tasks.append(SCALING_02_FILE_NAME)

    workspace.create_command_task(f"{DATA_20240729_SCALINGS_PREFIX}/all", all_tasks)

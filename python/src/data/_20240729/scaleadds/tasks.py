import os
from typing import Tuple

from matplotlib import pyplot, transforms, ticker

from data._20240729.constants import HOSHIHINA_600_FILE_NAME
from data._20240729.scaleadds.constants import DATA_20240729_SCALEADDS_PREFIX
from data._20240729.translations.tasks import get_hoshihina_image
from data._20240729.translations.test import plot_image
from pytasuku import Workspace


def plot_scaleadd_image(
        file_name: str,
        scale: Tuple[float, float],
        translation: Tuple[float, float],
        axis_x_lim=(-2,2), axis_y_lim=(-2,2)):
    fig, ((axis)) = pyplot.subplots(1, 1, figsize=(6,6))

    image = get_hoshihina_image()
    xform = transforms.Affine2D.from_values(scale[0], 0, 0, scale[1], translation[0], translation[1])
    plot_image(axis, image, transform=xform)

    axis.set_xlim(axis_x_lim[0], axis_x_lim[1])
    axis.set_ylim(axis_y_lim[0], axis_y_lim[1])

    axis.xaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))
    axis.yaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))

    axis.tick_params(axis='both', which='minor', length=0)   # remove minor tick lines

    axis.grid()

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    pyplot.savefig(file_name)
    pyplot.close(fig)


def define_data_20240729_scaleadds_tasks(workspace: Workspace):
    all_tasks = []

    x_axis_lim = (-5,5)
    y_axis_lim = (-5,5)

    ORIGINAL_FILE_NAME = f"{DATA_20240729_SCALEADDS_PREFIX}/hoshihina_original.png"
    workspace.create_file_task(
        ORIGINAL_FILE_NAME,
        [HOSHIHINA_600_FILE_NAME],
        lambda: plot_scaleadd_image(ORIGINAL_FILE_NAME,
                                    scale=(1.0,1.0),
                                    translation=(0,0),
                                    axis_x_lim=x_axis_lim,
                                    axis_y_lim=y_axis_lim))
    all_tasks.append(ORIGINAL_FILE_NAME)

    SCALEADD_00_FILE_NAME = f"{DATA_20240729_SCALEADDS_PREFIX}/hoshihina_scaleadd_00.png"
    workspace.create_file_task(
        SCALEADD_00_FILE_NAME,
        [HOSHIHINA_600_FILE_NAME],
        lambda: plot_scaleadd_image(SCALEADD_00_FILE_NAME,
                                    scale=(1.0,1.0),
                                    translation=(1,-1),
                                    axis_x_lim=x_axis_lim,
                                    axis_y_lim=y_axis_lim))
    all_tasks.append(SCALEADD_00_FILE_NAME)

    SCALEADD_01_FILE_NAME = f"{DATA_20240729_SCALEADDS_PREFIX}/hoshihina_scaleadd_01.png"
    workspace.create_file_task(
        SCALEADD_01_FILE_NAME,
        [HOSHIHINA_600_FILE_NAME],
        lambda: plot_scaleadd_image(SCALEADD_01_FILE_NAME,
                                    scale=(2.0,2.0),
                                    translation=(0,0),
                                    axis_x_lim=x_axis_lim,
                                    axis_y_lim=y_axis_lim))
    all_tasks.append(SCALEADD_01_FILE_NAME)

    SCALEADD_02_FILE_NAME = f"{DATA_20240729_SCALEADDS_PREFIX}/hoshihina_scaleadd_02.png"
    workspace.create_file_task(
        SCALEADD_02_FILE_NAME,
        [HOSHIHINA_600_FILE_NAME],
        lambda: plot_scaleadd_image(SCALEADD_02_FILE_NAME,
                                    scale=(2.0,2.0),
                                    translation=(1,-1),
                                    axis_x_lim=x_axis_lim,
                                    axis_y_lim=y_axis_lim))
    all_tasks.append(SCALEADD_02_FILE_NAME)

    SCALEADD_03_FILE_NAME = f"{DATA_20240729_SCALEADDS_PREFIX}/hoshihina_scaleadd_03.png"
    workspace.create_file_task(
        SCALEADD_03_FILE_NAME,
        [HOSHIHINA_600_FILE_NAME],
        lambda: plot_scaleadd_image(SCALEADD_03_FILE_NAME,
                                    scale=(2.0,2.0),
                                    translation=(2,-2),
                                    axis_x_lim=x_axis_lim,
                                    axis_y_lim=y_axis_lim))
    all_tasks.append(SCALEADD_03_FILE_NAME)

    workspace.create_command_task(f"{DATA_20240729_SCALEADDS_PREFIX}/all", all_tasks)
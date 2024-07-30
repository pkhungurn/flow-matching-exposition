import os

import numpy
import matplotlib
from matplotlib import pyplot
from matplotlib import transforms
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator

from data._20240729.constants import HOSHIHINA_600_FILE_NAME
from data._20240729.translations.constants import DATA_20240729_TRANSLATIONS_PREFIX
from pytasuku import Workspace, file_task
from shion.base.image_util import extract_numpy_image_from_filelike


def get_hoshihina_image():
    image = extract_numpy_image_from_filelike(HOSHIHINA_600_FILE_NAME)
    image = numpy.flip(image, axis=0)
    alpha = numpy.ones(shape=(image.shape[0], image.shape[1], 1))
    image = numpy.concatenate((image, alpha), axis=2)
    return image


def plot_image(ax, image, extent=(-1, 1, -1, 1), transform=None):
    im = ax.imshow(
        image,
        interpolation='antialiased',
        origin='lower',
        extent=extent,
        clip_on=True)

    if transform is not None:
        trans_data = transform + ax.transData
        im.set_transform(trans_data)


def plot_translated_image(
        file_name: str,
        x_dist: float,
        y_dist: float,
        axis_x_lim=(-2,2), axis_y_lim=(-2,2)):
    #matplotlib.rc('font', size=18)
    fig, ((axis)) = pyplot.subplots(1, 1, figsize=(6,6))

    image = get_hoshihina_image()
    xform = transforms.Affine2D().translate(x_dist, y_dist)
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


def define_data_20240729_translations_tasks(workspace: Workspace):
    all_tasks = []

    ORIGINAL_FILE_NAME = f"{DATA_20240729_TRANSLATIONS_PREFIX}/hoshihina_original.png"
    workspace.create_file_task(
        ORIGINAL_FILE_NAME,
        [HOSHIHINA_600_FILE_NAME],
        lambda: plot_translated_image(ORIGINAL_FILE_NAME, 0.0, 0.0))
    all_tasks.append(ORIGINAL_FILE_NAME)

    TRANSLATE_00_FILE_NAME = f"{DATA_20240729_TRANSLATIONS_PREFIX}/hoshihina_translate_00.png"
    workspace.create_file_task(
        TRANSLATE_00_FILE_NAME,
        [HOSHIHINA_600_FILE_NAME],
        lambda: plot_translated_image(TRANSLATE_00_FILE_NAME, 1.0, 0.0))
    all_tasks.append(TRANSLATE_00_FILE_NAME)

    TRANSLATE_01_FILE_NAME = f"{DATA_20240729_TRANSLATIONS_PREFIX}/hoshihina_translate_01.png"
    workspace.create_file_task(
        TRANSLATE_01_FILE_NAME,
        [HOSHIHINA_600_FILE_NAME],
        lambda: plot_translated_image(TRANSLATE_01_FILE_NAME, 0.0, -1.0))
    all_tasks.append(TRANSLATE_01_FILE_NAME)

    TRANSLATE_02_FILE_NAME = f"{DATA_20240729_TRANSLATIONS_PREFIX}/hoshihina_translate_02.png"
    workspace.create_file_task(
        TRANSLATE_02_FILE_NAME,
        [HOSHIHINA_600_FILE_NAME],
        lambda: plot_translated_image(TRANSLATE_02_FILE_NAME, -0.0, 0.0, axis_x_lim=(-7,2), axis_y_lim=(-2, 5)))
    all_tasks.append(TRANSLATE_02_FILE_NAME)

    TRANSLATE_03_FILE_NAME = f"{DATA_20240729_TRANSLATIONS_PREFIX}/hoshihina_translate_03.png"
    workspace.create_file_task(
        TRANSLATE_03_FILE_NAME,
        [HOSHIHINA_600_FILE_NAME],
        lambda: plot_translated_image(TRANSLATE_03_FILE_NAME, -5.0, 3.0, axis_x_lim=(-7,2), axis_y_lim=(-2, 5)))
    all_tasks.append(TRANSLATE_03_FILE_NAME)

    workspace.create_command_task(f"{DATA_20240729_TRANSLATIONS_PREFIX}/all", all_tasks)
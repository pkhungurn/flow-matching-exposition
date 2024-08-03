import os
from typing import Tuple

import numpy
from matplotlib import pyplot, ticker

from data._20240803.vector_fields.constants import DATA_20240803_VECTOR_FIELDS_PREFIX
from pytasuku import Workspace, file_task


def plot_vector_field(file_name: str,
                      vector_field: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray],
                      scale: float = 40,
                      color: str = 'blue',
                      axis_x_lim=(-3, 3),
                      axis_y_lim=(-3, 3),
                      title = None):
    fig, ((axis)) = pyplot.subplots(1, 1, figsize=(6, 6))

    X, Y, U, V = vector_field
    pyplot.quiver(X, Y, U, V, scale=scale, color=color)

    axis.set_xlim(axis_x_lim[0], axis_x_lim[1])
    axis.set_ylim(axis_y_lim[0], axis_y_lim[1])

    axis.xaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))
    axis.yaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))

    axis.tick_params(axis='both', which='minor', length=0)  # remove minor tick lines

    axis.grid()

    if title is not None:
        pyplot.title(title)

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    pyplot.savefig(file_name)
    pyplot.close(fig)


def define_data_20240803_vector_fields_tasks(workspace: Workspace):
    all_tasks = []

    vector_field_00_file_name = f"{DATA_20240803_VECTOR_FIELDS_PREFIX}/_00.png"
    @file_task(workspace, vector_field_00_file_name, [])
    def create_vector_field_00():
        x = numpy.linspace(-5, 5, 23)
        y = numpy.linspace(-5, 5, 23)
        X, Y = numpy.meshgrid(x, y)
        U = 0*X + 1
        V = 0*Y
        plot_vector_field(
            vector_field_00_file_name,
            (X,Y,U,V),
            scale=20)
    all_tasks.append(vector_field_00_file_name)

    vector_field_01_file_name = f"{DATA_20240803_VECTOR_FIELDS_PREFIX}/_01.png"
    @file_task(workspace, vector_field_01_file_name, [])
    def create_vector_field_01():
        x = numpy.linspace(-5, 5, 23)
        y = numpy.linspace(-5, 5, 23)
        X, Y = numpy.meshgrid(x, y)
        U = -Y
        V = X
        plot_vector_field(
            vector_field_01_file_name,
            (X,Y,U,V),
            scale=20)
    all_tasks.append(vector_field_01_file_name)

    vector_field_02_file_name = f"{DATA_20240803_VECTOR_FIELDS_PREFIX}/_02.png"
    @file_task(workspace, vector_field_02_file_name, [])
    def create_vector_field_01():
        x = numpy.linspace(-5, 5, 23)
        y = numpy.linspace(-5, 5, 23)
        X, Y = numpy.meshgrid(x, y)
        U = -X / numpy.sqrt(X**2 + Y**2 + 1e-8)
        V = -Y / numpy.sqrt(X**2 + Y**2 + 1e-8)
        plot_vector_field(
            vector_field_02_file_name,
            (X,Y,U,V),
            scale=20)
    all_tasks.append(vector_field_02_file_name)

    workspace.create_command_task(f"{DATA_20240803_VECTOR_FIELDS_PREFIX}/all", all_tasks)

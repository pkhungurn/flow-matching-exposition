import logging
import math
import os.path
from typing import Callable, Tuple

import numpy
from matplotlib import pyplot, ticker, transforms
from numpy import ndarray

from data._20240729.translations.tasks import get_hoshihina_image
from data._20240802.video_tasks import VideoTasksArgs
from data._20240803.tdvf.constants import DATA_20240803_TDVF_PREFIX
from data._20240803.vector_fields.tasks import plot_vector_field
from pytasuku import Workspace, file_task
from pytasuku.indexed.util import write_done_file


class TimeDependentVectorFieldTasksArgs:
    def __init__(self,
                 prefix: str,
                 num_frames: int,
                 vector_field_func: Callable[[float], Tuple[ndarray, ndarray, ndarray, ndarray]],
                 flow_func: Callable[[float, ndarray], ndarray],
                 points: ndarray,
                 axis_x_lim=(-3, 3),
                 axis_y_lim=(-3, 3),
                 scale=20.0):
        self.points = points
        self.flow_func = flow_func
        self.scale = scale
        self.vector_field_func = vector_field_func
        self.axis_y_lim = axis_y_lim
        self.axis_x_lim = axis_x_lim
        self.num_frames = num_frames
        self.prefix = prefix

    def frame_file_name(self, index):
        return f"{self.prefix}/frames/%08d.png" % index

    def frame_file_pattern(self):
        return f"{self.prefix}/frames/%08d.png"

    def frames_done_file_name(self):
        return f"{self.prefix}/frames_done.txt"

    def create_frame(self, index: int):
        if os.path.exists(self.frame_file_name(index)):
            return
        t = index * 1.0 / (self.num_frames - 1)
        X, Y, U, V = self.vector_field_func(t)
        plot_vector_field(self.frame_file_name(index), (X, Y, U, V), scale=self.scale, title=f"t = {'%0.2f' % t}")
        logging.info(f"Saved {self.frame_file_name(index)}")

    def frame_with_points_pattern(self):
        return f"{self.prefix}/frames_with_points/%08d.png"

    def frame_with_points_file_name(self, index: int):
        return f"{self.prefix}/frames_with_points/%08d.png" % index

    def frames_with_points_done_file_name(self):
        return f"{self.prefix}/frames_with_points_done.txt"

    def create_frame_with_points(self, index: int):
        if os.path.exists(self.frame_with_points_file_name(index)):
            return
        t = index * 1.0 / (self.num_frames - 1)
        X, Y, U, V = self.vector_field_func(t)

        fig, ((axis)) = pyplot.subplots(1, 1, figsize=(6, 6))

        pyplot.quiver(X, Y, U, V, scale=self.scale, color='blue')

        axis.set_xlim(self.axis_x_lim[0], self.axis_x_lim[1])
        axis.set_ylim(self.axis_y_lim[0], self.axis_y_lim[1])

        axis.xaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))
        axis.yaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))

        axis.tick_params(axis='both', which='minor', length=0)  # remove minor tick lines

        axis.grid()

        pyplot.title(f"t = {'%0.2f' % t}")

        points = self.flow_func(t, self.points)
        for i in range(points.shape[0]):
            x = points[i,0]
            y = points[i,1]
            circle = pyplot.Circle((x,y), 0.05, color='r')
            axis.add_patch(circle)

        file_name = self.frame_with_points_file_name(index)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        pyplot.savefig(file_name)
        pyplot.close(fig)

        logging.info(f"Saved {self.frame_file_name(index)}")

    def frame_with_image_pattern(self):
        return f"{self.prefix}/frames_with_image/%08d.png"

    def frame_with_image_file_name(self, index: int):
        return f"{self.prefix}/frames_with_image/%08d.png" % index

    def frames_with_image_done_file_name(self):
        return f"{self.prefix}/frames_with_image_done.txt"

    def create_frame_with_image(self, index):
        if os.path.exists(self.frame_with_image_file_name(index)):
            return
        t = index * 1.0 / (self.num_frames - 1)
        #X, Y, U, V = self.vector_field_func(t)

        fig, ((axis)) = pyplot.subplots(1, 1, figsize=(6, 6))

        #pyplot.quiver(X, Y, U, V, scale=self.scale, color='blue')

        axis.set_xlim(self.axis_x_lim[0], self.axis_x_lim[1])
        axis.set_ylim(self.axis_y_lim[0], self.axis_y_lim[1])

        axis.xaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))
        axis.yaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))

        axis.tick_params(axis='both', which='minor', length=0)  # remove minor tick lines

        axis.grid()

        pyplot.title(f"t = {'%0.2f' % t}")

        image = get_hoshihina_image()
        d = self.flow_func(t, numpy.array([[0.0,0.0]]))
        xform = transforms.Affine2D().translate(d[0,0], d[0,1])
        image = axis.imshow(
            image,
            interpolation='antialiased',
            origin='lower',
            extent=[-1,1,-1,1],
            clip_on=True)
        trans_data = xform + axis.transData
        image.set_transform(trans_data)

        file_name = self.frame_with_image_file_name(index)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        pyplot.savefig(file_name)
        pyplot.close(fig)

        logging.info(f"Saved {self.frame_file_name(index)}")

    def all_command_name(self):
        return f"{self.prefix}/all"

    def define_tasks(self, workspace: Workspace):
        all_tasks = []

        @file_task(workspace, self.frames_done_file_name(), [])
        def create_frames():
            for i in range(self.num_frames):
                self.create_frame(i)
            write_done_file(self.frames_done_file_name())

        all_tasks.append(self.frames_done_file_name())

        video_args = VideoTasksArgs(
            f"{self.prefix}/frame_video",
            self.frame_file_pattern(),
            self.num_frames,
            dependencies=[self.frames_done_file_name()])
        video_args.define_tasks(workspace)
        all_tasks.append(video_args.all_command_name())

        @file_task(workspace, self.frames_with_points_done_file_name(), [])
        def create_frames_with_points():
            for i in range(self.num_frames):
                self.create_frame_with_points(i)
            write_done_file(self.frames_with_points_done_file_name())
        all_tasks.append(self.frames_with_points_done_file_name())

        video_args = VideoTasksArgs(
            f"{self.prefix}/frames_with_points_video",
            self.frame_with_points_pattern(),
            self.num_frames,
            dependencies=[self.frames_with_points_done_file_name()])
        video_args.define_tasks(workspace)
        all_tasks.append(video_args.all_command_name())

        @file_task(workspace, self.frames_with_image_done_file_name(), [])
        def create_frames_with_image():
            for i in range(self.num_frames):
                self.create_frame_with_image(i)
            write_done_file(self.frames_with_image_done_file_name())
        all_tasks.append(self.frames_with_image_done_file_name())

        video_args = VideoTasksArgs(
            f"{self.prefix}/frames_with_image_video",
            self.frame_with_image_pattern(),
            self.num_frames,
            dependencies=[self.frames_with_image_done_file_name()])
        video_args.define_tasks(workspace)
        all_tasks.append(video_args.all_command_name())

        workspace.create_command_task(self.all_command_name(), all_tasks)


def define_data_20240803_tdvf_tasks(workspace: Workspace):
    all_tasks = []

    def tdvf_00(t: float):
        x = numpy.linspace(-5, 5, 23)
        y = numpy.linspace(-5, 5, 23)
        X, Y = numpy.meshgrid(x, y)
        U = numpy.cos(t * 2 * numpy.pi * 10) + 0 * X
        V = 0 * X
        return X, Y, U, V

    def flow_func_00(t: float, x: ndarray):
        sin_t = numpy.sin(t * 2 * numpy.pi * 10)
        translation = numpy.array([[sin_t, 0.0]])
        return x + translation

    args = TimeDependentVectorFieldTasksArgs(
        f"{DATA_20240803_TDVF_PREFIX}/_00",
        301,
        tdvf_00,
        flow_func_00,
        points=numpy.array([
            [0.0, 0.0],
            [2.0, 2.0],
            [-1.0, 1.0],
            [-1.5, -2.5],
            [0.5, -1.0]
        ]))
    args.define_tasks(workspace)
    all_tasks.append(args.all_command_name())

    workspace.create_command_task(f"{DATA_20240803_TDVF_PREFIX}/all", all_tasks)

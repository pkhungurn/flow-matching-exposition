import logging
import math
import os.path
from typing import Callable, Tuple, List, Optional

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
                 scale_func: Callable[[float], float],
                 translate_func: Callable[[float], Tuple[float, float]],
                 points: ndarray,
                 point_colors: Optional[List[str]],
                 axis_x_lim=(-3, 3),
                 axis_y_lim=(-3, 3),
                 scale=20.0):
        self.point_colors = point_colors
        self.translate_func = translate_func
        self.scale_func = scale_func
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

    def draw_grid(self, axis):
        axis.set_xlim(self.axis_x_lim[0], self.axis_x_lim[1])
        axis.set_ylim(self.axis_y_lim[0], self.axis_y_lim[1])

        axis.xaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))
        axis.yaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))

        axis.tick_params(axis='both', which='minor', length=0)  # remove minor tick lines

        axis.grid()

    def draw_vector_field(self, index: int):
        t = index * 1.0 / (self.num_frames - 1)
        X, Y, U, V = self.vector_field_func(t)
        pyplot.quiver(X, Y, U, V, scale=self.scale, color='blue')


    def draw_points(self, index: int, axis):
        t = index * 1.0 / (self.num_frames - 1)
        num_points = self.points.shape[0]
        points = self.flow_func(t, self.points)
        for i in range(num_points):
            x = points[i,0]
            y = points[i,1]
            if self.point_colors is not None:
                color = self.point_colors[i]
            else:
                color = 'r'
            circle = pyplot.Circle((x,y), 0.05, color=color)
            axis.add_patch(circle)

        xx = [[] for i in range(num_points)]
        yy = [[] for i in range(num_points)]
        for i in range(0, index+1):
            t = i * 1.0 / (self.num_frames-1)
            pp = self.flow_func(t, self.points)
            for j in range(num_points):
                xx[j].append(pp[j,0])
                yy[j].append(pp[j,1])
        for i in range(num_points):
            if self.point_colors is not None:
                color = self.point_colors[i]
            else:
                color = 'r'
            pyplot.plot(xx[i], yy[i], color=color)


    def create_frame_with_points(self, index: int):
        if os.path.exists(self.frame_with_points_file_name(index)):
            return
        t = index * 1.0 / (self.num_frames - 1)

        fig, ((axis)) = pyplot.subplots(1, 1, figsize=(6, 6))

        self.draw_grid(axis)
        self.draw_vector_field(index)
        self.draw_points(index, axis)
        pyplot.title(f"t = {'%0.2f' % t}")

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

    def draw_image(self, index:int, axis):
        t = index * 1.0 / (self.num_frames - 1)
        image = get_hoshihina_image()
        s = self.scale_func(t)
        d_x, d_y = self.translate_func(t)
        xform = transforms.Affine2D.from_values(s, 0, 0, s, d_x, d_y)
        image = axis.imshow(
            image,
            interpolation='antialiased',
            origin='lower',
            extent=[-1,1,-1,1],
            clip_on=True)
        trans_data = xform + axis.transData
        image.set_transform(trans_data)

    def create_frame_with_image(self, index):
        if os.path.exists(self.frame_with_image_file_name(index)):
            return
        t = index * 1.0 / (self.num_frames - 1)

        fig, ((axis)) = pyplot.subplots(1, 1, figsize=(6, 6))

        self.draw_grid(axis)
        self.draw_image(index, axis)
        self.draw_points(index,axis)
        pyplot.title(f"t = {'%0.2f' % t}")

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

    point_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
    ]

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
        scale_func=lambda t: 1.0,
        translate_func=lambda t: (numpy.sin(t * 2 * numpy.pi * 10), 0),
        points=numpy.array([
            [0.0, 0.0],
            [2.0, 2.0],
            [-1.0, 1.0],
            [-1.5, -2.5],
            [0.5, -1.0]
        ]),
        point_colors=point_colors)
    args.define_tasks(workspace)
    all_tasks.append(args.all_command_name())

    def tdvf_01(t: float):
        x = numpy.linspace(-5, 5, 23)
        y = numpy.linspace(-5, 5, 23)
        X, Y = numpy.meshgrid(x, y)
        U = (- 1.8*t) / (1-0.9*t**2) * (X + 2*t**2) - 4*t
        V = (- 1.8*t) / (1-0.9*t**2) * (Y - 1*t**2) + 2*t
        return X, Y, U, V

    def flow_func_01(t: float, x: ndarray):
        y = (1 - 0.9*t**2) * x
        y[:, 0] = y[:, 0] - 2*t**2
        y[:, 1] = y[:, 1] + t**2
        return y

    args = TimeDependentVectorFieldTasksArgs(
        f"{DATA_20240803_TDVF_PREFIX}/_01",
        101,
        tdvf_01,
        flow_func_01,
        scale_func=lambda t: 1.0-0.9*t**2,
        translate_func=lambda t: (-2*t**2, t**2),
        points=numpy.array([
            [0.0, 0.0],
            [2.0, 2.0],
            [-1.0, 1.0],
            [-1.5, -2.5],
            [0.5, -1.0]
        ]),
        scale=300,
        point_colors=point_colors)
    args.define_tasks(workspace)
    all_tasks.append(args.all_command_name())

    def tdvf_02(t: float):
        x = numpy.linspace(-5, 5, 23)
        y = numpy.linspace(-5, 5, 23)
        X, Y = numpy.meshgrid(x, y)

        x_data = -2
        y_data = 1
        angle = numpy.pi * (1-0.999*t) / 2
        sin_angle = numpy.sin(angle)
        cos_angle = numpy.cos(angle)

        U = -cos_angle / sin_angle * (X - x_data * cos_angle) + 0.999 * numpy.pi / 2 * sin_angle * x_data
        V = -cos_angle / sin_angle * (Y - y_data * cos_angle) + 0.999 * numpy.pi / 2 * sin_angle * y_data

        return X, Y, U, V

    def flow_func_02(t: float, x: ndarray):
        x_data = -2
        y_data = 1
        angle = numpy.pi * (1-0.999*t) / 2
        sin_angle = numpy.sin(angle)
        cos_angle = numpy.cos(angle)

        y = sin_angle * x
        y[:,0] += x_data * cos_angle
        y[:,1] += y_data * cos_angle

        return y

    def scale_func_02(t: float):
        angle = numpy.pi * (1-0.999*t) / 2
        sin_angle = numpy.sin(angle)
        return sin_angle

    def translate_func_02(t: float):
        x_data = -2
        y_data = 1
        angle = numpy.pi * (1-0.999*t) / 2
        cos_angle = numpy.cos(angle)
        return x_data*cos_angle, y_data*cos_angle

    args = TimeDependentVectorFieldTasksArgs(
        f"{DATA_20240803_TDVF_PREFIX}/_02",
        101,
        tdvf_02,
        flow_func_02,
        scale_func=scale_func_02,
        translate_func=translate_func_02,
        points=numpy.array([
            [0.0, 0.0],
            [2.0, 2.0],
            [-1.0, 1.0],
            [-1.5, -2.5],
            [0.5, -1.0]
        ]),
        scale=300,
        point_colors=point_colors)
    args.define_tasks(workspace)
    all_tasks.append(args.all_command_name())

    def tdvf_03(t: float):
        x = numpy.linspace(-5, 5, 23)
        y = numpy.linspace(-5, 5, 23)
        X, Y = numpy.meshgrid(x, y)

        x_data = -2
        y_data = 1

        U = (x_data - 0.999 *X) / (1 - 0.999*t)
        V = (y_data - 0.999 *Y) / (1 - 0.999*t)

        return X, Y, U, V

    def flow_func_03(t: float, x: ndarray):
        x_data = -2
        y_data = 1
        return (1.0-t)*x + t*(0.001 * x + numpy.array([[x_data, y_data]]))

    def scale_func_03(t: float):
        return 1 - 0.999*t

    def translate_func_03(t: float):
        x_data = -2
        y_data = 1
        return t*x_data, t*y_data

    args = TimeDependentVectorFieldTasksArgs(
        f"{DATA_20240803_TDVF_PREFIX}/_03",
        101,
        tdvf_03,
        flow_func_03,
        scale_func=scale_func_03,
        translate_func=translate_func_03,
        points=numpy.array([
            [0.0, 0.0],
            [2.0, 2.0],
            [-1.0, 1.0],
            [-1.5, -2.5],
            [0.5, -1.0]
        ]),
        scale=300,
        point_colors=point_colors)
    args.define_tasks(workspace)
    all_tasks.append(args.all_command_name())

    workspace.create_command_task(f"{DATA_20240803_TDVF_PREFIX}/all", all_tasks)

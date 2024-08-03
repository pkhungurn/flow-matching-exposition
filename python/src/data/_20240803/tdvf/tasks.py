import logging
import math
import os.path
from typing import Callable, Tuple

import numpy
from numpy import ndarray

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
                 axis_x_lim=(-3, 3),
                 axis_y_lim=(-3, 3),
                 scale=20.0):
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
        X,Y,U,V = self.vector_field_func(t)
        plot_vector_field(self.frame_file_name(index), (X,Y,U,V), scale=self.scale, title=f"t = {'%0.2f' % t}")
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

        workspace.create_command_task(self.all_command_name(), all_tasks)


def define_data_20240803_tdvf_tasks(workspace: Workspace):
    all_tasks = []

    def tdvf_00(t: float):
        x = numpy.linspace(-5, 5, 23)
        y = numpy.linspace(-5, 5, 23)
        X, Y = numpy.meshgrid(x, y)
        U = numpy.cos(t * 2 * numpy.pi * 10) + 0*X
        V = 0*X
        return X, Y, U, V

    args = TimeDependentVectorFieldTasksArgs(
        f"{DATA_20240803_TDVF_PREFIX}/_00",
        301,
        tdvf_00)
    args.define_tasks(workspace)
    all_tasks.append(args.all_command_name())

    workspace.create_command_task(f"{DATA_20240803_TDVF_PREFIX}/all", all_tasks)

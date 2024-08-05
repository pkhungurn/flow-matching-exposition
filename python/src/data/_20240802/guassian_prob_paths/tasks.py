import os
from typing import Callable, Tuple

import numpy
from matplotlib import pyplot, ticker

from data._20240729.probdist.tasks import get_gaussian_probdist
from data._20240802.guassian_prob_paths.constants import DATA_20240802_GAUSSIAN_PROB_PATHS_PREFIX
from data._20240802.video_tasks import VideoTasksArgs
from pytasuku import Workspace, file_task
from pytasuku.indexed.util import write_done_file


class GaussianProbabilityPathTasksArgs:
    def __init__(self,
                 prefix: str,
                 num_frames: int,
                 mu_func: Callable[[numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]],
                 sigma_func: Callable[[numpy.ndarray], numpy.ndarray],
                 axis_x_lim=(-3, 3),
                 axis_y_lim=(-3, 3),
                 v_min=0.0,
                 v_max=1.0):
        self.v_max = v_max
        self.v_min = v_min
        self.axis_y_lim = axis_y_lim
        self.axis_x_lim = axis_x_lim
        self.sigma_func = sigma_func
        self.mu_func = mu_func
        self.num_frames = num_frames
        self.prefix = prefix

    def sigma_plot_file_name(self):
        return f"{self.prefix}/sigma_plot.png"

    def mu_plot_file_name(self):
        return f"{self.prefix}/mu_plot.png"

    def gaussian_viz_frame_file_pattern(self):
        return f"{self.prefix}/gaussian_viz_frames/%08d.png"

    def gaussian_viz_frame_file_name(self, index: int):
        return f"{self.prefix}/gaussian_viz_frames/{'%08d' % index}.png"

    def gaussian_viz_frames_done_file_name(self):
        return f"{self.prefix}/gaussian_viz_frames_done.txt"

    def create_mu_plot(self):
        os.makedirs(self.prefix, exist_ok=True)

        t = numpy.linspace(0.0, 1.0, self.num_frames)
        x, y = self.mu_func(t)

        fig, ((axis)) = pyplot.subplots(1, 1, figsize=(6, 6))

        axis.set_xlim(self.axis_x_lim[0], self.axis_x_lim[1])
        axis.set_ylim(self.axis_y_lim[0], self.axis_y_lim[1])

        axis.xaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))
        axis.yaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))

        axis.tick_params(axis='both', which='minor', length=0)  # remove minor tick lines

        axis.grid()

        pyplot.title('Mean ($\\mu_t$) Trajectory')
        pyplot.xlabel('$x^1$')
        pyplot.ylabel('$x^2$')
        pyplot.plot(x, y, color='orange')
        pyplot.savefig(self.mu_plot_file_name())
        pyplot.close()

    def create_sigma_plot(self):
        os.makedirs(self.prefix, exist_ok=True)

        t = numpy.linspace(0.0, 1.0, self.num_frames)
        sigma_t = self.sigma_func(t)
        pyplot.figure(figsize=(6,6))
        pyplot.title("Standard Deviation ($\\sigma_t$)")
        pyplot.xlabel('$t$')
        pyplot.ylabel('$\\sigma_t$')
        pyplot.plot(t, sigma_t)
        pyplot.savefig(self.sigma_plot_file_name())
        pyplot.close()

    def create_gaussian_viz_frame(self, index: int):
        t = index / (self.num_frames - 1)
        mu_x, mu_y = self.mu_func(numpy.array([t]))
        sigma = self.sigma_func(numpy.array([t]))

        fig, ((axis)) = pyplot.subplots(1, 1, figsize=(6, 6))

        p = get_gaussian_probdist(mu_x[0], mu_y[0], sigma[0])

        axis.imshow(
            p,
            interpolation='antialiased',
            origin='lower',
            extent=[self.axis_x_lim[0], self.axis_x_lim[1], self.axis_y_lim[0], self.axis_y_lim[1]],
            vmin=self.v_min,
            vmax=self.v_max,
            clip_on=True)

        axis.set_xlim(self.axis_x_lim[0], self.axis_x_lim[1])
        axis.set_ylim(self.axis_y_lim[0], self.axis_y_lim[1])

        axis.xaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))
        axis.yaxis.set(major_locator=ticker.MultipleLocator(1), minor_locator=ticker.MultipleLocator(0.1))

        axis.tick_params(axis='both', which='minor', length=0)  # remove minor tick lines

        axis.grid()

        tt = numpy.linspace(0.0, 1.0, self.num_frames)
        xx, yy = self.mu_func(tt)
        pyplot.plot(xx, yy, color='orange')

        circle = pyplot.Circle((mu_x[0], mu_y[0]), sigma[0], color='r', fill=False)
        axis.add_patch(circle)

        circle = pyplot.Circle((mu_x[0], mu_y[0]), 0.05, color='r')
        axis.add_patch(circle)

        pyplot.title(
            f"$t = {'%0.2f' % t}, \\mu_t = ({'%0.2f' % mu_x[0]}, {'%0.2f' % mu_y[0]}), \\sigma_t = {'%0.2f' % sigma[0]}$")

        file_name = self.gaussian_viz_frame_file_name(index)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        pyplot.savefig(file_name)
        pyplot.close(fig)

    def all_task_name(self):
        return f"{self.prefix}/all"

    def define_tasks(self, workspace: Workspace):
        all_tasks = []

        workspace.create_file_task(self.sigma_plot_file_name(), [], self.create_sigma_plot)
        all_tasks.append(self.sigma_plot_file_name())

        workspace.create_file_task(self.mu_plot_file_name(), [], self.create_mu_plot)
        all_tasks.append(self.mu_plot_file_name())

        @file_task(workspace, self.gaussian_viz_frames_done_file_name(), [])
        def create_gaussian_viz_frames():
            for i in range(self.num_frames):
                if os.path.exists(self.gaussian_viz_frame_file_name(i)):
                    continue
                self.create_gaussian_viz_frame(i)
            write_done_file(self.gaussian_viz_frames_done_file_name())

        all_tasks.append(self.gaussian_viz_frames_done_file_name())

        video_args = VideoTasksArgs(
            f"{self.prefix}/gaussian_viz_video",
            self.gaussian_viz_frame_file_pattern(),
            self.num_frames,
            dependencies=[
                self.gaussian_viz_frames_done_file_name()])
        video_args.define_tasks(workspace)
        all_tasks.append(video_args.all_command_name())

        workspace.create_command_task(self.all_task_name(), all_tasks)


def define_data_20240802_gaussian_prob_paths_tasks(workspace: Workspace):
    all_tasks = []

    def sigma_00(t: numpy.ndarray):
        return 1.0 - 0.5 * t ** 2

    def mu_00(t: numpy.ndarray):
        theta = numpy.pi / 2 * (1.0 - t)
        x = 2 * numpy.cos(theta)
        y = 2 * (numpy.sin(theta) - 1.0)
        return x, y

    args = GaussianProbabilityPathTasksArgs(
        f"{DATA_20240802_GAUSSIAN_PROB_PATHS_PREFIX}/path_00",
        101,
        mu_func=mu_00,
        sigma_func=sigma_00,
        v_max=1.0 / (2.0 * numpy.pi * 0.5))
    args.define_tasks(workspace)
    all_tasks.append(args.all_task_name())

    def sigma_01(t: numpy.ndarray):
        return 1.0 * (1.0 - t) + 0.5 * t

    def mu_01(t: numpy.ndarray):
        x = t * 2
        y = -t * 2
        return x, y

    args = GaussianProbabilityPathTasksArgs(
        f"{DATA_20240802_GAUSSIAN_PROB_PATHS_PREFIX}/path_01",
        101,
        mu_func=mu_01,
        sigma_func=sigma_01,
        v_max=1.0 / (2.0 * numpy.pi * 0.5))
    args.define_tasks(workspace)
    all_tasks.append(args.all_task_name())

    def sigma_02(t: numpy.ndarray):
        return 1.0 + 0.0 * t

    def mu_02(t: numpy.ndarray):
        x = numpy.sin(2 * 10 * numpy.pi * t)
        y = 0.0 * t
        return x, y

    args = GaussianProbabilityPathTasksArgs(
        f"{DATA_20240802_GAUSSIAN_PROB_PATHS_PREFIX}/path_02",
        301,
        mu_func=mu_02,
        sigma_func=sigma_02,
        v_max=1.0 / (2.0 * numpy.pi))
    args.define_tasks(workspace)
    all_tasks.append(args.all_task_name())

    def sigma_03(t: numpy.ndarray):
        return 1 - 0.9*t**2

    def mu_03(t: numpy.ndarray):
        x = -2*t**2
        y = t**2
        return x, y

    args = GaussianProbabilityPathTasksArgs(
        f"{DATA_20240802_GAUSSIAN_PROB_PATHS_PREFIX}/path_03",
        101,
        mu_func=mu_03,
        sigma_func=sigma_03,
        v_max=1.0 / (2.0 * numpy.pi))
    args.define_tasks(workspace)
    all_tasks.append(args.all_task_name())

    workspace.create_command_task(f"{DATA_20240802_GAUSSIAN_PROB_PATHS_PREFIX}/all", all_tasks)

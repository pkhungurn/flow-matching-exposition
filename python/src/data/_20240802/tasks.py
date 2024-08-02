from data._20240802.constants import DATA_20240802_PREFIX
from data._20240802.gaussian_viz.constants import DATA_20240802_GAUSSIAN_VIZ_PREFIX
from data._20240802.gaussian_viz.tasks import define_data_20240802_gaussian_viz_tasks
from data._20240802.guassian_prob_paths.constants import DATA_20240802_GAUSSIAN_PROB_PATHS_PREFIX
from data._20240802.guassian_prob_paths.tasks import define_data_20240802_gaussian_prob_paths_tasks
from pytasuku import Workspace


def define_data_20240802_tasks(workspace: Workspace):
    define_data_20240802_gaussian_viz_tasks(workspace)
    define_data_20240802_gaussian_prob_paths_tasks(workspace)

    workspace.create_command_task(
        f"{DATA_20240802_PREFIX}/all",
        [
            f"{DATA_20240802_GAUSSIAN_VIZ_PREFIX}/all",
            f"{DATA_20240802_GAUSSIAN_PROB_PATHS_PREFIX}/all",
        ])

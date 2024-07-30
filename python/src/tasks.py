from data._20240729.tasks import define_data_20240729_tasks
from pytasuku import Workspace
from slides.tasks import define_slides_tasks


def define_tasks(workspace: Workspace):
    define_data_20240729_tasks(workspace)

    define_slides_tasks(workspace)
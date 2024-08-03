from data._20240803.constants import DATA_20240803_PREFIX
from data._20240803.tdvf.constants import DATA_20240803_TDVF_PREFIX
from data._20240803.tdvf.tasks import define_data_20240803_tdvf_tasks
from data._20240803.vector_fields.constants import DATA_20240803_VECTOR_FIELDS_PREFIX
from data._20240803.vector_fields.tasks import define_data_20240803_vector_fields_tasks
from pytasuku import Workspace


def define_data_20240803_tasks(workspace: Workspace):
    define_data_20240803_vector_fields_tasks(workspace)
    define_data_20240803_tdvf_tasks(workspace)

    workspace.create_command_task(
        f"{DATA_20240803_PREFIX}/all",
        [
            f"{DATA_20240803_VECTOR_FIELDS_PREFIX}/all",
            f"{DATA_20240803_TDVF_PREFIX}/all",
        ])
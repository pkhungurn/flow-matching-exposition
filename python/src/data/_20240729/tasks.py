from data._20240729.constants import DATA_20240729_PREFIX
from data._20240729.probdist.constants import DATA_20240729_PROBDIST_PREFIX
from data._20240729.probdist.tasks import define_data_20240729_probdist_tasks
from data._20240729.scaleadds.constants import DATA_20240729_SCALEADDS_PREFIX
from data._20240729.scaleadds.tasks import define_data_20240729_scaleadds_tasks
from data._20240729.scalings.constants import DATA_20240729_SCALINGS_PREFIX
from data._20240729.scalings.tasks import define_data_20240729_scalings_tasks
from data._20240729.translations.constants import DATA_20240729_TRANSLATIONS_PREFIX
from data._20240729.translations.tasks import define_data_20240729_translations_tasks
from pytasuku import Workspace


def define_data_20240729_tasks(workspace: Workspace):
    define_data_20240729_translations_tasks(workspace)
    define_data_20240729_scalings_tasks(workspace)
    define_data_20240729_scaleadds_tasks(workspace)
    define_data_20240729_probdist_tasks(workspace)

    workspace.create_command_task(
        f"{DATA_20240729_PREFIX}/all",
        [
            f"{DATA_20240729_TRANSLATIONS_PREFIX}/all",
            f"{DATA_20240729_SCALINGS_PREFIX}/all",
            f"{DATA_20240729_SCALEADDS_PREFIX}/all",
            f"{DATA_20240729_PROBDIST_PREFIX}/all",
        ])

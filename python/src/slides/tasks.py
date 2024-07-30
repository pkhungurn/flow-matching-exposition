import os
import shutil

from data._20240729.translations.constants import DATA_20240729_TRANSLATIONS_PREFIX
from pytasuku import Workspace


def copy_file(source_file_name: str, dest_file_name: str):
    os.makedirs(os.path.dirname(dest_file_name), exist_ok=True)
    shutil.copyfile(source_file_name, dest_file_name)


def define_slides_tasks(workspace: Workspace):
    all_tasks = []

    if False:
        task = workspace.create_file_task(
            f"slides/_0002/images/hoshihina_original.png",
            [f"{DATA_20240729_TRANSLATIONS_PREFIX}/hoshihina_original.png"],
            lambda: copy_file(
                f"{DATA_20240729_TRANSLATIONS_PREFIX}/hoshihina_original.png",
                f"slides/_0002/images/hoshihina_original.png"))
        all_tasks.append(task.name)

        task = workspace.create_file_task(
            f"slides/_0005/images/hoshihina_translate_00.png",
            [f"{DATA_20240729_TRANSLATIONS_PREFIX}/hoshihina_translate_00.png"],
            lambda: copy_file(
                f"{DATA_20240729_TRANSLATIONS_PREFIX}/hoshihina_translate_00.png",
                f"slides/_0005/images/hoshihina_translate_00.png"))
        all_tasks.append(task.name)

    workspace.create_command_task(f"slides/copy_files", all_tasks)
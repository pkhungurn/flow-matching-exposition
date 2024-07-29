import logging
import os
import sys

import tasks
from pytasuku import *


def replace_sep_with_slash(path):
    comps = path.split(os.path.sep)
    return "/".join(comps)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python src/run.py <task-name-1> <task-name-2> ...")
        sys.exit(0)

    logging.basicConfig(level=logging.INFO, force=True)
    workspace = Workspace()
    tasks.define_tasks(workspace)

    workspace.start_session()
    for arg in sys.argv[1:]:
        arg = replace_sep_with_slash(arg)
        workspace.run(arg)
    workspace.end_session()

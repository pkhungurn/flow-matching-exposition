import logging
import os
import sys
from typing import Optional


def get_torchrun_executable():
    return os.path.dirname(sys.executable) + os.path.sep + "torchrun"


def run_distributed_training_script(
        training_script_file_name: str,
        num_nodes: int,
        node_rank: int,
        num_proc_per_node: int,
        master_addr: int = "127.0.0.1",
        master_port: int = 8888):
    command = f"{get_torchrun_executable()} " \
              f"--nproc_per_node={num_proc_per_node} " \
              f"--nnodes={num_nodes} " \
              f"--node_rank={node_rank} " \
              f"--master_addr={master_addr} " \
              f"--master_port={master_port} " \
              f"{training_script_file_name}"
    logging.info(f"Executing -- {command}")
    os.system(command)


class RdzvConfig:
    def __init__(self, id: int, port: int):
        self.port = port
        self.id = id


def run_standalone_distributed_training_script(
        training_script_file_name: str,
        num_proc_per_node: int,
        target_checkpoint_examples: Optional[int] = None,
        rdzv_config: Optional[RdzvConfig] = None):
    command = f"{get_torchrun_executable()} " \
              f"--nnodes=1 " \
              f"--nproc_per_node={num_proc_per_node} "
    if rdzv_config is not None:
        command += f"--rdzv_endpoint=localhost:{rdzv_config.port} "
        command += "--rdzv_backend=c10d "
        command += f"--rdzv_id={rdzv_config.id} "
    else:
        command += "--standalone "
    command += f"{training_script_file_name} "
    if target_checkpoint_examples is not None:
        command += f"--target_checkpoint_examples {target_checkpoint_examples} "
    logging.info(f"Executing -- {command}")
    os.system(command)

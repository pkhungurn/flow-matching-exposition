import os
import sys
from typing import Callable, List, Optional

from pytasuku import Workspace
from shion.core.training.distrib.distributed_trainer import DistributedTrainer
from shion.core.training.distrib.distributed_training_script_util import run_distributed_training_script, RdzvConfig, \
    run_standalone_distributed_training_script
from shion.core.training.distrib.distributed_training_states import DistributedTrainingState


def define_distributed_training_tasks(
        workspace: Workspace,
        prefix: str,
        training_script_file_name: str,
        num_nodes: int,
        num_proc_per_node: int,
        master_addr: int = "127.0.0.1",
        master_port: int = 8888):
    def run_training_script_func(rank: int):
        def _f():
            run_distributed_training_script(
                training_script_file_name,
                num_nodes,
                rank,
                num_proc_per_node, master_addr,
                master_port)

        return _f

    for i in range(num_nodes):
        workspace.create_command_task(f"{prefix}/train_node_%06d" % i, [], run_training_script_func(i))


def define_standalone_distributed_training_tasks(
        workspace: Workspace,
        distributed_trainer_func: Callable[[int], DistributedTrainer],
        training_script_file_name: str,
        num_proc_per_node: int,
        dependencies: Optional[List[str]] = None,
        rdzv_config: Optional[RdzvConfig] = None):
    trainer = distributed_trainer_func(1)
    checkpoint_examples = trainer.training_protocol.get_checkpoint_examples()
    assert len(checkpoint_examples) >= 1
    assert checkpoint_examples[0] > 0
    checkpoint_examples = [0] + checkpoint_examples

    if dependencies is None:
        dependencies = []
    module_file_dependencies = dependencies[:]
    for module_name in trainer.pretrained_module_file_names:
        module_file_dependencies.append(trainer.pretrained_module_file_names[module_name])

    def create_train_func(target_checkpoint_examples: int):
        return lambda: run_standalone_distributed_training_script(
            training_script_file_name,
            num_proc_per_node,
            target_checkpoint_examples,
            rdzv_config=rdzv_config)

    train_tasks = []
    for checkpoint_index in range(0, len(checkpoint_examples)):
        for module_name in trainer.module_names:
            module_file_name = DistributedTrainingState.get_module_file_name(
                trainer.get_checkpoint_prefix(checkpoint_index),
                module_name)
            workspace.create_file_task(
                module_file_name,
                module_file_dependencies,
                create_train_func(trainer.checkpoint_examples[checkpoint_index]))
        for module_name in trainer.accumulators:
            accumulated_module_file_name = DistributedTrainingState.get_accumulated_module_file_name(
                trainer.get_checkpoint_prefix(checkpoint_index),
                module_name)
            workspace.create_file_task(
                accumulated_module_file_name,
                module_file_dependencies,
                create_train_func(checkpoint_examples[checkpoint_index]))
        workspace.create_command_task(
            trainer.get_checkpoint_prefix(checkpoint_index) + "/train_standalone",
            module_file_dependencies,
            create_train_func(checkpoint_examples[checkpoint_index]))
        train_tasks.append(trainer.get_checkpoint_prefix(checkpoint_index) + "/train_standlone")
    workspace.create_file_task(
        trainer.prefix + "/train_standalone",
        module_file_dependencies,
        create_train_func(checkpoint_examples[-1]))


if __name__ == "__main__":
    print(os.path.dirname(sys.executable) + os.path.sep + "torchrun")

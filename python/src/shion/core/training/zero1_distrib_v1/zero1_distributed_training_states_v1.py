import copy
import logging
import os
from typing import Dict, Optional, Callable

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel

from shion.core.load_save import torch_save, torch_load
from shion.core.module_accumulator import ModuleAccumulator
from shion.core.module_factory import ModuleFactory
from shion.core.optimizer_factory import OptimizerFactory


def print_peak_memory(prefix, device):
    print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


class Zero1DistributedTrainingStateV1:
    def __init__(self,
                 examples_seen_so_far: int,
                 modules: Dict[str, Module],
                 accumulated_modules: Dict[str, Module],
                 optimizers: Dict[str, ZeroRedundancyOptimizer]):
        self.accumulated_modules = accumulated_modules
        self.optimizers = optimizers
        self.modules = modules
        self.examples_seen_so_far = examples_seen_so_far

    @staticmethod
    def get_examples_seen_so_far_file_name(prefix) -> str:
        return prefix + "/examples_seen_so_far.txt"

    @staticmethod
    def get_module_file_name(prefix, module_name) -> str:
        return "%s/module_%s.pt" % (prefix, module_name)

    @staticmethod
    def get_accumulated_module_file_name(prefix, module_name) -> str:
        return "%s/accumulated_%s.pt" % (prefix, module_name)

    @staticmethod
    def get_optimizer_file_name(prefix, module_name) -> str:
        return "%s/optimizer_%s.pt" % (prefix, module_name)

    @staticmethod
    def get_rng_state_file_name(prefix, rank: int):
        return "%s/rng_state_%08d.pt" % (prefix, rank)

    def mkdir(self, prefix: str):
        os.makedirs(prefix, exist_ok=True)

    def save_data(self, prefix: str, rank: int):
        assert os.path.exists(prefix)

        torch_save(torch.get_rng_state(), Zero1DistributedTrainingStateV1.get_rng_state_file_name(prefix, rank))
        logging.info("Saved %s" % Zero1DistributedTrainingStateV1.get_rng_state_file_name(prefix, rank))

        if rank == 0:
            logging.info("Saving training state to %s" % prefix)
            with open(Zero1DistributedTrainingStateV1.get_examples_seen_so_far_file_name(prefix), "wt") as fout:
                fout.write("%d\n" % self.examples_seen_so_far)
                logging.info("Saved %s" % Zero1DistributedTrainingStateV1.get_examples_seen_so_far_file_name(prefix))
            for module_name in self.modules:
                if module_name not in self.optimizers:
                    continue
                file_name = Zero1DistributedTrainingStateV1.get_module_file_name(prefix, module_name)
                module = self.modules[module_name]
                if isinstance(module, DistributedDataParallel):
                    state_dict = module.module.state_dict()
                else:
                    state_dict = module.state_dict()
                torch_save(state_dict, file_name)
                logging.info("Saved %s" % file_name)
            for module_name in self.accumulated_modules:
                file_name = Zero1DistributedTrainingStateV1.get_accumulated_module_file_name(prefix, module_name)
                torch_save(self.accumulated_modules[module_name].state_dict(), file_name)
                logging.info("Saved %s" % file_name)
            for module_name in self.optimizers:
                file_name = Zero1DistributedTrainingStateV1.get_optimizer_file_name(prefix, module_name)
                torch_save(self.optimizers[module_name].state_dict(), file_name)
                logging.info("Saved %s" % file_name)

        logging.info("Done saving training state to %s" % prefix)

    def save(self, prefix: str, rank: int, barrier_func: Callable[[], None]):
        if rank == 0:
            self.mkdir(prefix)
        barrier_func()

        # Consolidate optimizer states
        for module_name in self.optimizers:
            optimizer = self.optimizers[module_name]
            optimizer.consolidate_state_dict(0)

        barrier_func()

        self.save_data(prefix, rank)

        barrier_func()

        # Clear optimizer consolidated states.
        for module_name in self.optimizers:
            optimizer = self.optimizers[module_name]
            optimizer._all_state_dicts = []

    @staticmethod
    def get_examples_seen_so_far(prefix: str) -> int:
        with open(Zero1DistributedTrainingStateV1.get_examples_seen_so_far_file_name(prefix)) as fin:
            lines = fin.readlines()
            return int(lines[0])

    @staticmethod
    def load(
            prefix: str,
            module_factories: Dict[str, ModuleFactory],
            accumulators: Dict[str, ModuleAccumulator],
            optimizer_factories: Dict[str, OptimizerFactory],
            rank: int,
            local_rank: int,
            device: torch.device,
            pretrained_module_file_names: Optional[Dict[str, str]] = None) -> 'Zero1DistributedTrainingStateV1':
        if pretrained_module_file_names is None:
            pretrained_module_file_names = {}

        logging.info(f"[Rank {rank}] Loading training state from {prefix}")

        with open(Zero1DistributedTrainingStateV1.get_examples_seen_so_far_file_name(prefix)) as fin:
            lines = fin.readlines()
            examples_seen_so_far = int(lines[0])
            logging.info(
                f"[Rank {rank}] Loaded {Zero1DistributedTrainingStateV1.get_examples_seen_so_far_file_name(prefix)}")

        modules = {
            module_name: factory.create()
            for (module_name, factory) in module_factories.items()
        }
        for module_name in modules:
            if module_name in optimizer_factories:
                file_name = Zero1DistributedTrainingStateV1.get_module_file_name(prefix, module_name)
            else:
                assert module_name in pretrained_module_file_names
                file_name = pretrained_module_file_names[module_name]
            module = modules[module_name]
            state_dict = torch_load(file_name)
            module.load_state_dict(state_dict)
            module.to(device)
            modules[module_name] = DistributedDataParallel(
                module,
                device_ids=[device.index],
                output_device=device.index)
            logging.info(f"[Rank {rank}] Loaded module '{module_name}' from {file_name}")

        #print_peak_memory(f"[rank={rank}] Max memory allocated after loading models", rank)

        accumulated_modules = {}
        for module_name in accumulators:
            module_factory = module_factories[module_name]
            module = module_factory.create()
            file_name = Zero1DistributedTrainingStateV1.get_accumulated_module_file_name(prefix, module_name)
            module.load_state_dict(torch_load(file_name))
            module.to(device)
            accumulated_modules[module_name] = module
            logging.info(f"[Rank {rank}] Loaded {file_name}")

        #print_peak_memory(f"[rank={rank}] Max memory allocated after loading accumulated model", rank)

        optimizers = {}
        for module_name in optimizer_factories:
            file_name = Zero1DistributedTrainingStateV1.get_optimizer_file_name(prefix, module_name)
            module = modules[module_name]
            optimizer = ZeroRedundancyOptimizer(
                module.parameters(),
                optimizer_class=optimizer_factories[module_name].get_optimizer_class(),
                **optimizer_factories[module_name].get_optimizer_hyperparameters())
            optimizer.load_state_dict(torch_load(file_name))
            optimizers[module_name] = optimizer
            logging.info(f"[Rank {rank}] Loaded {file_name}")

            #print(rank, len(optimizer.optim.param_groups[0]['params']))
            #print(rank, optimizer._partition_parameters_cache[0][0].keys())

        #print_peak_memory(f"[rank={rank}] Max memory allocated after loading optimizers", rank)

        torch.set_rng_state(torch_load(Zero1DistributedTrainingStateV1.get_rng_state_file_name(prefix, rank)))
        logging.info(
            f"[Rank {rank}] Loaded {Zero1DistributedTrainingStateV1.get_examples_seen_so_far_file_name(prefix)}")

        logging.info(f"[Rank {rank}] Done loading training state from {prefix}")

        return Zero1DistributedTrainingStateV1(examples_seen_so_far, modules, accumulated_modules, optimizers)

    @staticmethod
    def new(module_factories: Dict[str, ModuleFactory],
            accumulators: Dict[str, ModuleAccumulator],
            optimizer_factories: Dict[str, OptimizerFactory],
            random_seed: int,
            rank: int,
            local_rank: int,
            device: torch.device,
            pretrained_module_file_names: Optional[Dict[str, str]] = None) -> 'Zero1DistributedTrainingStateV1':
        examples_seen_so_far = 0

        modules = {
            module_name: factory.create()
            for (module_name, factory) in module_factories.items()
        }
        for module_name in modules:
            modules[module_name].to(device)
        if pretrained_module_file_names is not None:
            for module_name in modules:
                if module_name in pretrained_module_file_names:
                    file_name = pretrained_module_file_names[module_name]
                    modules[module_name].load_state_dict(torch_load(file_name))
                    logging.info(f"Loaded initial state of '{module_name}' from {file_name} ...")

        accumulated_modules = {}
        for module_name in accumulators:
            accumulated_modules[module_name] = copy.deepcopy(modules[module_name])

        for module_name in modules:
            module = modules[module_name]
            modules[module_name] = DistributedDataParallel(
                module,
                device_ids=[device.index],
                output_device=device.index)

        optimizers = {}
        for module_name in optimizer_factories:
            module = modules[module_name]
            optimizer = ZeroRedundancyOptimizer(
                module.parameters(),
                optimizer_class=optimizer_factories[module_name].get_optimizer_class(),
                **optimizer_factories[module_name].get_optimizer_hyperparameters())
            optimizers[module_name] = optimizer

        torch.manual_seed(random_seed + rank)

        return Zero1DistributedTrainingStateV1(examples_seen_so_far, modules, accumulated_modules, optimizers)

    @staticmethod
    def can_load(prefix: str,
                 module_factories: Dict[str, ModuleFactory],
                 accumulators: Dict[str, ModuleAccumulator],
                 optimizer_factories: Dict[str, OptimizerFactory],
                 pretrained_module_file_names: Dict[str, str],
                 world_size: int) -> bool:
        logging.info(f"Checking directory {prefix}")
        if not os.path.isdir(prefix):
            logging.info(f"Cannot load files in {prefix} because it is not a directory")
            return False
        examples_seen_so_far_file_name = Zero1DistributedTrainingStateV1.get_examples_seen_so_far_file_name(prefix)
        if not os.path.isfile(examples_seen_so_far_file_name):
            logging.info(f"Cannot load files in {prefix} because {examples_seen_so_far_file_name} is not a file.")
            return False
        for module_name in module_factories.keys():
            if module_name in optimizer_factories:
                file_name = Zero1DistributedTrainingStateV1.get_module_file_name(prefix, module_name)
                if not os.path.isfile(file_name):
                    logging.info(f"Cannot load files in {prefix} because {file_name} is not a file.")
                    return False
            else:
                if not module_name in pretrained_module_file_names:
                    return False
                if not os.path.isfile(pretrained_module_file_names[module_name]):
                    return False
        for module_name in accumulators:
            file_name = Zero1DistributedTrainingStateV1.get_accumulated_module_file_name(prefix, module_name)
            if not os.path.isfile(file_name):
                logging.info(f"Cannot load files in {prefix} because {file_name} is not a file.")
                return False
        for module_name in optimizer_factories:
            file_name = Zero1DistributedTrainingStateV1.get_optimizer_file_name(prefix, module_name)
            if not os.path.isfile(file_name):
                logging.info(f"Cannot load files in {prefix} because {file_name} is not a file.")
                return False
        for rank in range(world_size):
            file_name = Zero1DistributedTrainingStateV1.get_rng_state_file_name(prefix, rank)
            if not os.path.isfile(file_name):
                logging.info(f"Cannot load files in {prefix} because {file_name} is not a file.")
                return False
        return True

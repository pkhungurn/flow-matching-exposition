from abc import ABC, abstractmethod
from typing import Iterable, Type, Any, Dict

from torch.nn import Parameter
from torch.optim.optimizer import Optimizer


class OptimizerFactory(ABC):
    @abstractmethod
    def create(self, parameters: Iterable[Parameter]):
        pass

    @abstractmethod
    def get_optimizer_class(self) -> Type[Optimizer]:
        pass

    @abstractmethod
    def get_optimizer_hyperparameters(self) -> Dict[str, Any]:
        pass
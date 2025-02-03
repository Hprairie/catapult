from abc import ABCMeta, abstractmethod, abstractclassmethod
from typing import List


class BackendDriver(metaclass=ABCMeta):

    @abstractclassmethod
    def is_active() -> bool:
        pass

    @abstractmethod
    def get_target() -> str:
        pass

    def __init__(self) -> None:
        pass


class GPUDriver(BackendDriver):

    @abstractmethod
    def get_device(self) -> str:
        pass

    @abstractmethod
    def set_device(self, device: str) -> None:
        pass

    @abstractmethod
    def get_stream(self) -> str:
        pass

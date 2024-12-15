from typing import Tuple, Any
from abc import ABCMeta, abstractmethod

from catapult.runtime import types


class Compiler(metaclass=ABCMeta):
    _template_conversions = {
        types.int1: lambda arg: str(arg),
        types.int8: lambda arg: str(arg),
        types.int16: lambda arg: str(arg),
        types.int32: lambda arg: str(arg),
        types.int64: lambda arg: str(arg),
        types.float16: lambda arg: str(arg),
        types.float32: lambda arg: str(arg),
        types.float64: lambda arg: str(arg),
        types.bfloat16: lambda arg: str(arg),
        types.uint8: lambda arg: str(arg),
        types.uint16: lambda arg: str(arg),
        types.uint32: lambda arg: str(arg),
        types.uint64: lambda arg: str(arg),
        types.void: lambda arg: str(arg),
        int: lambda arg: str(arg),
        float: lambda arg: str(arg),
        str: lambda arg: str(arg),
        bool: lambda arg: str(arg).lower(),
    }

    _special_kernel_kwargs = ["stream", "smem"]

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __del__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_source(self) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def compile(self, template_vals) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_kernel(self) -> Tuple[Any, Any]:
        raise NotImplementedError

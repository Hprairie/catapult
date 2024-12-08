from abc import ABCMeta, abstractmethod


class Compiler(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None: ...

    @abstractmethod
    def __del__() -> None: ...

    @abstractmethod
    def get_source() -> str: ...

    @abstractmethod
    def get_name() -> str: ...

    @abstractmethod
    def compile() -> None: ...

    @abstractmethod
    def add_named_expression() -> None: ...

    @abstractmethod
    def get_kernel() -> None: ...

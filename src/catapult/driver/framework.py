from abc import ABCMeta, abstractmethod
from typing import List, Tuple


class Framework(metaclass=ABCMeta):
    """Base abstract class for handling different computation frameworks.

    This class provides an interface for framework-specific operations like
    data pointer parsing and device management. Implementations should handle
    framework-specific details (e.g., PyTorch, NumPy) while providing a
    consistent interface for kernel compilation and execution.
    """

    @staticmethod
    @abstractmethod
    def is_active() -> bool:
        """Check if the framework is available in the current environment.

        Returns:
            bool: True if the framework is available, False otherwise.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_available_targets() -> List[str]:
        """Get the target platform/backend for the framework.

        Returns:
            str: Identifier for the target platform (e.g., 'cpu').
        """
        raise NotImplementedError

    @abstractmethod
    def set_target(self, target: str) -> None:
        """Set the target platform/backend for the framework.

        Args:
            target (str): Identifier for the target platform (e.g., 'cpu').
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Get the name of the framework.

        Returns:
            str: Name of the framework (e.g., 'torch', 'numpy').
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def clean_values(args) -> Tuple[Tuple, Tuple]:
        """Clean the Framework specific values in the input to easily parse and pass to the backend."""
        raise NotImplementedError


class GPUFramework(Framework):
    """Abstract base class for GPU-specific framework implementations.

    Extends the Framework class with GPU-specific operations like device
    management and stream handling. Used for frameworks that support GPU
    computation (e.g., PyTorch CUDA, CuPy).
    """

    @staticmethod
    @abstractmethod
    def get_device() -> int:
        """Get the current GPU device identifier.

        Returns:
            str: Current device identifier.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def set_device(device: str) -> None:
        """Set the current GPU device.

        Args:
            device (str): Device identifier to set as current.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_stream() -> int:
        """Get the current CUDA stream.

        Returns:
            str: Current CUDA stream identifier.
        """
        raise NotImplementedError

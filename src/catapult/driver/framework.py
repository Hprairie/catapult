from abc import ABCMeta, abstractmethod
from typing import List


class Framework(metaclass=ABCMeta):
    """Base abstract class for handling different computation frameworks.
    
    This class provides an interface for framework-specific operations like
    data pointer parsing and device management. Implementations should handle
    framework-specific details (e.g., PyTorch, NumPy) while providing a 
    consistent interface for kernel compilation and execution.
    """

    @classmethod
    @abstractmethod
    def is_active() -> bool:
        """Check if the framework is available in the current environment.
        
        Returns:
            bool: True if the framework is available, False otherwise.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_available_targets() -> str:
        """Get the target platform/backend for the framework.
        
        Returns:
            str: Identifier for the target platform (e.g., 'cpu').
        """
        pass

    @abstractmethod
    def set_target() -> None:
        """Set the target platform/backend for the framework.
        
        Args:
            target (str): Identifier for the target platform (e.g., 'cpu').
        """
    
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Get the name of the framework.
        
        Returns:
            str: Name of the framework (e.g., 'torch', 'numpy').
        """
    
    @staticmethod
    @abstractmethod
    def clean_values() -> None:
        """Clean the Framework specific values in the input to easily parse and pass to the backend."""
        pass


class GPUFramework(Framework):
    """Abstract base class for GPU-specific framework implementations.
    
    Extends the Framework class with GPU-specific operations like device
    management and stream handling. Used for frameworks that support GPU
    computation (e.g., PyTorch CUDA, CuPy).
    """

    @staticmethod
    @abstractmethod
    def get_device(self) -> str:
        """Get the current GPU device identifier.
        
        Returns:
            str: Current device identifier.
        """
        pass

    @staticmethod
    @abstractmethod
    def set_device(self, device: str) -> None:
        """Set the current GPU device.
        
        Args:
            device (str): Device identifier to set as current.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_stream() -> str:
        """Get the current CUDA stream.
        
        Returns:
            str: Current CUDA stream identifier.
        """
        pass

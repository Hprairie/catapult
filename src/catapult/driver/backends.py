from abc import ABCMeta, abstractmethod

from catapult.compiler import Compiler


class Backend(metaclass=ABCMeta):
    """Abstract base class for backend-specific kernel compiler implementations.

    This class defines the interface for backend-specific compiler wrappers.
    Implementations should handle the compilation of kernel code for specific
    hardware backends (e.g., CUDA, OpenCL, etc.).

    Each backend implementation should provide methods to:
    1. Create a compiler instance for a specific kernel
    2. Check if the backend is available on the system
    3. Launch the kernel on the backend
    """

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the backend compiler."""

    @staticmethod
    @abstractmethod
    def get_compiler() -> Compiler:
        """Create and return a compiler object for this backend.

        Returns:
            Compiler: A compiler instance specific to this backend
                     (e.g., _NVRTCProgram for CUDA)
        """

    @abstractmethod
    def launch_backend() -> None:
        """Launch the backend."""

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Check if this backend is available on the current system.

        Returns:
            bool: True if the backend is available, False otherwise
        """

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Get the name of the backend.

        Returns:
            str: Name of the backend (e.g., 'cuda', 'opencl')
        """

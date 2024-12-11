import ctypes
from catapult.driver import Framework, GPUFramework

try:
    import torch
except ImportError:
    torch = None


class TorchBaseFramework(Framework):
    def __init__(self) -> None:
        raise NotImplementedError


class TorchGPUFramework(GPUFramework):

    def __init__(self) -> None:

        # TODO: Check if there is a better place to initialize CUDA
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        
        # Initialize target to None
        # Requires that we register a target backend before using the framework
        self.target = None

    def get_stream(self, idx) -> str:
        return torch.cuda.current_stream(idx).cuda_stream
    
    def get_device(self, device: int | str):
        return torch.cuda.current_device(device)
    
    def set_device(self, device: int | str):
        return torch.cuda.set_device(device)

    @staticmethod
    def is_active() -> bool:
        return torch is not None

    @staticmethod
    def get_name() -> str:
        return "torch"
    
    def get_available_targets():
        return ["cuda"]
    
    def set_target(self, target: str) -> None:
        self.target = target
    
    @staticmethod
    def clean_values(args):
        """
        Prepares arguments for CUDA kernel launch with proper C types using a dictionary-based approach.

        Args:
            *args: Variable number of arguments of different types

        Returns:
            tuple: (arg_values, arg_types) for cuLaunchKernel
        """

        # Dictionary mapping Python types to their corresponding ctype handlers

        TYPE_HANDLERS = {
            torch.Tensor: lambda x: (x.data_ptr(), ctypes.c_void_p),
            bool: lambda x: (ctypes.c_bool(x), ctypes.c_bool),
            int: lambda x: (ctypes.c_size_t(x), ctypes.c_size_t),
            float: lambda x: (ctypes.c_float(x), ctypes.c_float),
        }

        def get_ctype_and_value(arg):
            # Look up the handler for this type
            handler = TYPE_HANDLERS.get(type(arg))
            if handler is None:
                raise TypeError(f"Unsupported argument type: {type(arg)}")
            return handler(arg)

        # Process all arguments
        processed_args = [get_ctype_and_value(arg) for arg in args]

        # Split into values and types
        arg_values = tuple(value for value, _ in processed_args)
        arg_types = tuple(type_ for _, type_ in processed_args)

        return arg_values, arg_types

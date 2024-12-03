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


# class TorchCUDADriver(GPUDriver):
    
#         def __init__(self) -> None:
#             try:
#                 import torch
#             except ImportError:
#                 # TODO: Get better error messaging
#                 raise ImportError("Torch is not installed.")
#             finally:
#                 self.torch = torch
            
#             self.get_device = self.torch.current_device
#             self.set_device = self.torch.cuda.set_device
            
    
#         def get_stream(self, idx) -> str:
#             return self.torch.cuda.current_stream(idx).cuda_stream
    
#         def is_active() -> bool:
#             return True
    
#         def get_target() -> str:
#             return "torch.cuda"

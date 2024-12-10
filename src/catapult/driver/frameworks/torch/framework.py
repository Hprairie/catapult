from catapult.driver import Framework, GPUFramework


class TorchBaseFramework(Framework):
    def __init__(self) -> None:
        raise NotImplementedError


class TorchGPUFramework(GPUFramework):

    _driver_type = "torch"

    def __init__(self) -> None:
        try:
            import torch
        except ImportError:
            torch = None
        finally:
            self.torch = torch

        if self.torch is not None:
            self.get_device = self.torch.current_device
            self.set_device = self.torch.cuda.set_device
        
        # TODO: Check if there is a better place to initialize CUDA
        if not torch.cuda.is_initialized():
            torch.cuda.init()

    def get_stream(self, idx) -> str:
        return self.torch.cuda.current_stream(idx).cuda_stream

    def is_active() -> bool:
        try:
            import torch

            return True if torch.cuda.is_available() else False
        except ImportError:
            return False

    def get_target(self) -> str:
        return self._driver_type

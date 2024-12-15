from functools import lru_cache
from typing import Optional

from catapult.driver import backends, frameworks, Driver

FRAMEWORKS = {
    "torch": frameworks["torch"].TorchGPUFramework,
}
BACKENDS = {
    "cuda": backends["cuda"].CUDABackend,
}


@lru_cache
def get_driver(framework_name: Optional[str] = None, backend_name: Optional[str] = None) -> Driver:
    # TODO: Actually create a driver which scans available tools and selects the best one
    if framework_name is None:
        framework_name = "torch"
    if backend_name is None:
        backend_name = "cuda"
    return Driver(FRAMEWORKS[framework_name](), BACKENDS[backend_name]())

from functools import lru_cache
from catapult.driver import backends, framework, Driver

FRAMEWORKS = {"torch": framework.TorchGPUFramework,}
BACKENDS = {"cuda": backends.CUDABackend,}

@lru_cache
def get_driver(framework_name: str | None, backend_name: str | None) -> Driver:
    # TODO: Actually create a driver which scans available tools and selects the best one
    if framework_name is None:
        framework_name = "torch"
    if backend_name is None:
        backend_name = "cuda"
    return Driver(FRAMEWORKS[framework_name](), BACKENDS[backend_name]())
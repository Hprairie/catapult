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
    if framework_name is not None and framework_name not in FRAMEWORKS:
        supported = ", ".join(sorted(FRAMEWORKS.keys()))
        raise ValueError(f"Framework '{framework_name}' is not supported. Available frameworks: {supported}")
    if backend_name is not None and backend_name not in BACKENDS:
        supported = ", ".join(sorted(BACKENDS.keys()))
        raise ValueError(f"Backend '{backend_name}' is not supported. Available backends: {supported}")
    if framework_name is None:
        framework_name = "torch"
    if backend_name is None:
        backend_name = "cuda"
    return Driver(FRAMEWORKS[framework_name](), BACKENDS[backend_name]())

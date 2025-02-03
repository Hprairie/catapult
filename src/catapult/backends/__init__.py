import os
import importlib
from dataclasses import dataclass
from .driver import BackendDriver


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class Backend:
    driver: BackendDriver
    # TODO: Potential to abstract out the compiler if we want to add CPU support


def _discover_backends():
    backends = {}
    root = os.path.dirname(__file__)
    for backend in os.listdir(root):
        driver = _load_module(backend, os.path.join(root, backend, "driver.py"))
        backends[backend] = Backend(driver)


backends = _discover_backends()

__all__ = [
    "BackendDriver",
    "backends",
]

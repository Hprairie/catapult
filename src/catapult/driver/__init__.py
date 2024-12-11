import os
import importlib
from .framework import Framework, GPUFramework
from .backends import Backend
from .driver import Driver


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def _discover_frameworks():
    frameworks = {}
    root = os.path.dirname(__file__)
    root_framework = os.path.join(root, "frameworks")
    for framework in os.listdir(root_framework):
        framework_module = _load_module(framework, os.path.join(root_framework, framework, "framework.py"))
        frameworks[framework] = framework_module
    
    return frameworks

def _discover_backends():
    backends = {}
    root = os.path.dirname(__file__)
    root_backends = os.path.join(root, "backends")
    for backend in os.listdir(root_backends):
        backend_module = _load_module(backend, os.path.join(root_backends, backend, "backend.py"))
        backends[backend] = backend_module
    
    return backends


backends = _discover_backends()
frameworks = _discover_frameworks()

__all__ = [
    "Framework",
    "GPUFramework",
    "Backend",
    "Driver",
    "backends",
    "frameworks",
]

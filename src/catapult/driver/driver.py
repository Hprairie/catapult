from .backends import Backend
from .framework import Framework, GPUFramework


class Driver:
    def __init__(self, framework: Framework | GPUFramework, backend: Backend) -> None:
        if not issubclass(type(framework), Framework) or not issubclass(type(framework), GPUFramework):
            raise TypeError(f"Expected a Framework when creating a Driver, but got {type(framework)}")
        if not issubclass(type(backend), Backend):
            raise TypeError(f"Expected a Backend when creating a Driver, but got {type(backend)}")
        if backend.get_name() not in framework.get_available_targets():
            raise TypeError(
                f"Expected a Backend with the same target as the Framework, but got {backend.get_name()} and {framework.get_available_targets()}"
            )
        self.framework = framework
        self.backend = backend

    def get_name(self) -> str:
        return self.framework.get_name() + "." + self.backend.get_name()

    def register_new_backend(self, backend: Backend) -> None:
        if not issubclass(type(backend), Backend):
            raise TypeError(f"Expected a Backend when creating a Driver, but got {type(backend)}")
        if self.framework.get_available_targets() != backend.get_name():
            raise TypeError(
                f"Expected a Backend with the same target as the Framework, but got {backend.get_name()} and {self.framework.get_available_targets}"
            )
        self.backend = backend

    def register_new_framework(self, framework: Framework) -> None:
        if not issubclass(type(framework), Framework):
            raise TypeError(f"Expected a Framework when creating a Driver, but got {type(framework)}")
        if self.backend.get_name() != framework.get_available_targets():
            raise TypeError(
                f"Expected a Framework with the same target as the Backend, but got {framework.get_available_targets()} and {self.backend.get_name()}"
            )
        self.framework = framework

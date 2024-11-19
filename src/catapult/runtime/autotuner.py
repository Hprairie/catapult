from typing import Dict, Callable, Any, Tuple, Optional, List


class Config:

    def __init__(
        self,
        kwargs: Dict[str, Any],
        thread_grid: Optional[Tuple[int]] = None,
        pre_hook: Optional[Callable] = None,
    ) -> None:
        self.kwargs = kwargs
        self.thread_grid = thread_grid
        self.pre_hook = pre_hook

    def get_kwargs(self): ...

    def __str__(self) -> str:
        output = []
        for key, val in self.kwargs.items():
            output.append(f"{key}={val}")
        output.append(f"thread_grid={self.thread_grid}")
        output.append(f"pre_hook={self.pre_hook}")
        return ", ".join(output)


class AutoTuner:

    def __init__(
        self,
        fn: Callable,
        configs: List[Config],
        key: Optional[List[str]] = None,
    ) -> None:
        self.fn = fn
        self.configs = configs
        self.key = key
        self.cache = {}
        if key is not None:
            raise NotImplementedError("NOT IMPLEMENTED")

    def _do_benchmark(self): ...

    def __call__(self, *args, **kwargs) -> Any: ...


def autotuner(configs, key) -> Callable:
    def decorator(fn):
        return AutoTuner(fn=fn, configs=configs, key=key)

    return decorator

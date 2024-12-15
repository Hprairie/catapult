import time
import torch
import os
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import functools


@dataclass
class Config:
    """Configuration container for kernel parameters"""

    params: Dict[str, Any]

    def __getitem__(self, key):
        return self.params[key]

    def items(self):
        return self.params.items()


class Autotuner:
    def __init__(
        self,
        configs: List[Config],
        key: List[str],
        prune_configs_by: Optional[Dict[str, Any]] = None,
        reset_to_zero: Optional[List[str]] = None,
        restore_value: Optional[List[str]] = None,
        warmup: int = 10,
        rep: int = 50,
    ):
        # TODO: Create better error message
        if not isinstance(configs, list):
            raise TypeError("configs must be a list of Config objects")
        if not isinstance(key, list):
            raise TypeError("key must be a list of strings")
        if not all(isinstance(c, Config) for c in configs):
            raise TypeError("configs must be a list of Config objects")
        if not all(isinstance(k, str) for k in key):
            raise TypeError("key must be a list of strings")
        if len(configs) == 0:
            raise ValueError("configs must not be empty")

        self.configs = configs
        self.key = key
        self.prune_configs_by = prune_configs_by or {}
        self.reset_to_zero = reset_to_zero or []
        self.restore_value = restore_value or []
        self.warmup = warmup
        self.rep = rep
        self.cache = {}

        if len(self.reset_to_zero):
            raise NotImplementedError

        if len(self.restore_value):
            raise NotImplementedError

        if len(self.prune_configs_by):
            raise NotImplementedError

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = tuple(kwargs.get(k) for k in self.key)
            if cache_key in self.cache:
                return func(*args, **{**kwargs, **self.cache[cache_key].params})

            best_time = float("inf")
            best_config = None

            # Benchmark each configuration
            timing_results = []
            print_results = os.environ.get("CATAPULT_PRINT_AUTOTUNING") == "1"

            for config in self.configs:
                tensor_args = args
                tensor_kwargs = kwargs

                # Add config parameters to kwargs
                test_kwargs = {**tensor_kwargs, **config.params}

                # Warmup runs
                for _ in range(self.warmup):
                    func(*tensor_args, **test_kwargs)
                torch.cuda.synchronize()

                # Benchmark runs
                start = time.perf_counter()
                for _ in range(self.rep):
                    func(*tensor_args, **test_kwargs)
                torch.cuda.synchronize()
                end = time.perf_counter()
                avg_time = (end - start) / self.rep

                if print_results:
                    timing_results.append((config, avg_time))

                if avg_time < best_time:
                    best_time = avg_time
                    best_config = config

            # TODO: Get better error handeling
            if best_config is None:
                raise RuntimeError("best_config is NONE when it shouldn't")

            if print_results:
                print(f"\nCATAPULT AUTOTUNING RESULTS FOR KERNEL: {func.__name__}")
                print("-" * 50)
                for config, time_val in sorted(timing_results, key=lambda x: x[1]):
                    print(f"Config: {config.params}")
                    print(f"Average time: {time_val*1000:.3f} ms")
                    print("-" * 50)
                print(f"Best config selected: {best_config.params}")
                print("-" * 50)
                print("\n")

            # Cache the best config
            self.cache[cache_key] = best_config

            # Return result with best config
            return func(*args, **{**kwargs, **best_config.params})

        return wrapper


# ---------------------------
# Autotune decorator
# ---------------------------


def autotune(configs: List[Config], key: List[str], **kwargs) -> Callable:
    return Autotuner(configs, key, **kwargs)

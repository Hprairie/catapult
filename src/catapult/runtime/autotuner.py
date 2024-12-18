import time
import torch
import os
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import functools


@dataclass
class Config:
    """Configuration container for kernel parameters

    Attributes:
        params: Dictionary containing kernel parameter names and their values
    """

    params: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.params[key]

    def items(self) -> List[tuple[str, Any]]:
        return list(self.params.items())


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
        if not isinstance(configs, list):
            raise TypeError(f"'configs' must be a list, got {type(configs).__name__}")
        if not isinstance(key, list):
            raise TypeError(f"'key' must be a list, got {type(key).__name__}")
        if not all(isinstance(c, Config) for c in configs):
            invalid_configs = [c for c in configs if not isinstance(c, Config)]
            raise TypeError(f"All items in 'configs' must be Config objects. Invalid items: {invalid_configs}")
        if not all(isinstance(k, str) for k in key):
            invalid_keys = [k for k in key if not isinstance(k, str)]
            raise TypeError(f"All items in 'key' must be strings. Invalid items: {invalid_keys}")
        if len(configs) == 0:
            raise ValueError("'configs' list cannot be empty - at least one configuration is required")

        self.configs = configs
        self.key = key
        self.prune_configs_by = prune_configs_by or {}
        self.reset_to_zero = reset_to_zero or []
        self.restore_value = restore_value or []
        self.warmup = warmup
        self.rep = rep
        self.cache = {}

        if len(self.reset_to_zero):
            raise NotImplementedError("reset_to_zero feature is not yet implemented")

        if len(self.restore_value):
            raise NotImplementedError("restore_value feature is not yet implemented")

        if len(self.prune_configs_by):
            raise NotImplementedError("prune_configs_by feature is not yet implemented")

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

            if best_config is None:
                raise RuntimeError(
                    f"Failed to find best configuration for kernel '{func.__name__}'. "
                    f"This could be caused by:\n"
                    f"1. All configurations failed to execute\n"
                    f"2. No valid timing measurements were obtained\n"
                    f"Current configs tested: {[c.params for c in self.configs]}\n"
                    f"This error is unexpected and should be reported."
                )

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


def autotune(
    configs: List[Config],
    key: List[str],
    *,
    prune_configs_by: Optional[Dict[str, Any]] = None,
    reset_to_zero: Optional[List[str]] = None,
    restore_value: Optional[List[str]] = None,
    warmup: int = 10,
    rep: int = 50,
) -> Callable:
    return Autotuner(
        configs=configs,
        key=key,
        prune_configs_by=prune_configs_by,
        reset_to_zero=reset_to_zero,
        restore_value=restore_value,
        warmup=warmup,
        rep=rep,
    )

import time
import torch
import os
from typing import List, Dict, Any, Callable, Optional, Union
from dataclasses import dataclass
import functools
import copy

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
        warmup: int = 1,
        rep: int = 3
    ):
        self.configs = configs
        self.key = key
        self.prune_configs_by = prune_configs_by or {}
        self.reset_to_zero = reset_to_zero or []
        self.restore_value = restore_value or []
        self.warmup = warmup
        self.rep = rep
        self.cache = {}
        
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from the key arguments
            cache_key = tuple(kwargs.get(k) for k in self.key)
            
            if cache_key in self.cache:
                return func(*args, **{**kwargs, **self.cache[cache_key].params})
            
            # Store original values for restoration
            original_values = {k: kwargs.get(k) for k in self.restore_value}
            
            # Reset specified arguments to zero
            for k in self.reset_to_zero:
                if k in kwargs:
                    kwargs[k] = 0
                    
            # Prune configs if performance model is provided
            configs_to_test = self.configs
            if 'perf_model' in self.prune_configs_by and 'top_k' in self.prune_configs_by:
                perf_model = self.prune_configs_by['perf_model']
                top_k = self.prune_configs_by['top_k']
                
                # Sort configs by predicted performance
                configs_with_perf = [(c, perf_model(c)) for c in configs_to_test]
                configs_to_test = [c for c, _ in sorted(configs_with_perf, key=lambda x: x[1])[:top_k]]
            
            # Early pruning if provided
            if 'early_config_prune' in self.prune_configs_by:
                configs_to_test = self.prune_configs_by['early_config_prune'](configs_to_test)
            
            best_time = float('inf')
            best_config = None
            
            # Benchmark each configuration
            timing_results = []
            print_results = os.environ.get('CATAPULT_PRINT_AUTOTUNING') == '1'
            
            for config in configs_to_test:
                # Create copies of tensor arguments to prevent in-place modifications
                tensor_args = [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]
                tensor_kwargs = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v 
                    for k, v in kwargs.items()
                }
                
                # Add config parameters to kwargs
                test_kwargs = {**tensor_kwargs, **config.params}
                
                # Warmup runs
                for _ in range(self.warmup):
                    func(*tensor_args, **test_kwargs)
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                # Benchmark runs
                for _ in range(self.rep):
                    func(*tensor_args, **test_kwargs)
                
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                avg_time = (end - start) / self.rep
                timing_results.append((config, avg_time))
                
                if avg_time < best_time:
                    best_time = avg_time
                    best_config = config
            
            if print_results:
                print("\nCATAPULT AUTOTUNING RESULTS:")
                print("-" * 50)
                for config, time_val in sorted(timing_results, key=lambda x: x[1]):
                    print(f"Config: {config.params}")
                    print(f"Average time: {time_val*1000:.3f} ms")
                    print("-" * 50)
                print(f"Best config selected: {best_config.params}")
                print("-" * 50)
                print("\n")
            
            # Restore original values
            for k, v in original_values.items():
                kwargs[k] = v
            
            # Cache the best config
            self.cache[cache_key] = best_config
            
            # Return result with best config
            return func(*args, **{**kwargs, **best_config.params})
        
        return wrapper

# Add to catapult namespace
def autotune(
    configs: List[Config],
    key: List[str],
    **kwargs
) -> Callable:
    return Autotuner(configs, key, **kwargs)
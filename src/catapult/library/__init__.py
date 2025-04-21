import os
import importlib.util
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

functions = {}

for item in os.listdir(current_dir):
    folder_path = os.path.join(current_dir, item)
    
    if os.path.isdir(folder_path):
        impl_file = os.path.join(folder_path, 'impl.py')
        
        if os.path.exists(impl_file):
            module_name = f"catapult.library.{item}.impl"
            spec = importlib.util.spec_from_file_location(module_name, impl_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            if hasattr(module, 'func'):
                functions[item] = module.func  # Using just the folder name as the key

__all__ = ['functions']

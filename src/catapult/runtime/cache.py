import os
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict
from catapult.runtime.errors import CacheError


def get_cache_path():
    return os.getenv("CATAPULT_CACHE", Path.home())


def get_cache_dir():
    return os.path.join(get_cache_path(), ".catapult", "cache")


class TemplateCacheManager(ABC):
    def __init__(self, key) -> None:
        pass

    @abstractmethod
    def get_file(self, filename) -> Optional[str]:
        pass

    @abstractmethod
    def get_group(self, filenames) -> Optional[Dict[str, str]]:
        pass

    @abstractmethod
    def put(self, filenames) -> Optional[str]:
        pass

    @abstractmethod
    def put_group(self, filenames) -> Optional[Dict[str, str]]:
        pass


class LocalCacheManager(TemplateCacheManager):
    def __init__(self, key) -> None:
        self.key = key

    def get_file(self, filename) -> Optional[str]:
        pass

    def get_group(self, filenames) -> Optional[Dict[str, str]]:
        pass

    def put(self, filenames) -> Optional[str]:
        pass

    def put_group(self, filenames) -> Optional[Dict[str, str]]:
        pass


def get_cached_ptx(source, kernel_name):
    """
    Get's the cached PTX between runs. Will return None if it doesn't exists or the hash is not the same.

    Current Issues:
        - Will silently fail if the header files are different
    """
    if CACHE_DIR is None:
        raise CacheError(
            f"Error when accessing kernel cache. The environment variable CATAPULT_CACHE is not set. Please reinstall catapult."
        )
    if not os.path.isdir(CACHE_DIR):
        raise CacheError(
            f"Error when accessing kernel cache. The environment variable CATAPULT_CACHE is set to {CACHE_DIR}"
        )
    cached_kernel_hash_path = os.path.join(CACHE_DIR, f"{kernel_name}.hash")

    if not os.path.isfile(cached_kernel_hash_path):
        return None

    with open(cached_kernel_hash_path, "r") as f:
        cached_kernel_hash = f.read().strip()

    current_kernel_hash = _get_source_hash(source)
    if current_kernel_hash == cached_kernel_hash:
        cached_kernel_ptx_path = os.path.join(CACHE_DIR, f"{kernel_name}.ptx")
        with open(cached_kernel_ptx_path, "rb") as f:
            return f.read()

    # Return None if not an exact match (will overwrite)
    return None


def save_cached_ptx(ptx, kernel_name):
    """
    Save's the cached PTX between runs. Will overwrite the current cached .ptx and .hash files if they already exist for the given kernel.
    """
    if CACHE_DIR is None:
        raise CacheError(
            f"Error when creating kernel cache. The environment variable CATAPULT_CACHE is not set. Please reinstall catapult."
        )
    if not os.path.isdir(CACHE_DIR):
        raise CacheError(
            f"Error when creating kernel cache. The environment variable CATAPULT_CACHE is set to {CACHE_DIR}"
        )
    cached_kernel_hash_path = os.path.join(CACHE_DIR, f"{kernel_name}.hash")
    cached_kernel_ptx_path = os.path.join(CACHE_DIR, f"{kernel_name}.ptx")

    source_hash = _get_source_hash(ptx)


def _get_source_hash(source):
    if os.path.isfile(source):
        with open(source, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    elif isinstance(source, str):
        return hashlib.sha256(source.encode()).hexdigest()
    else:
        raise ValueError(f"Attempting to create a cached ptx with a source type of {type(source)}")

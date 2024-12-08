import os
from typing import Optional, Tuple, List

from ...backends import Backend
from ....compiler import _NVRTCProgram


class CUDABackend(Backend):
    def __init__(self) -> None:
        pass

    def get_compiler(
        source: str | bytes,
        name: str | bytes,
        calling_dir: str,
        num_headers: int = 0,
        headers: Optional[Tuple[bytes] | List[bytes]] = None,
        include_names: Optional[Tuple[bytes] | List[bytes]] = None,
        method: str = "ptx",
    ) -> _NVRTCProgram:
        if isinstance(source, str) and os.path.isfile(os.path.join(calling_dir, source)):
            with open(os.path.join(calling_dir, source), "r") as f:
                source = f.read()
        if isinstance(source, str):
            source = bytes(source, "utf-8")
        if isinstance(name, str):
            name = bytes(name, "utf-8")
        return _NVRTCProgram(
            source=source,
            name=name,
            num_headers=num_headers,
            headers=headers,
            include_names=include_names,
            method=method,
        )

    @classmethod
    def is_available() -> bool:
        raise NotImplementedError

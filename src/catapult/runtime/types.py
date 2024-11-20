from typing import List, Optional


class dtype:
    SINT_TYPES = ["int8", "int16", "int32", "int64"]
    UINT_TYPES = ["int1", "uint8", "uint16", "uint32", "uint64"]
    FP_TYPES = ["fp16", "bf16", "fp32", "fp64"]
    OTHER_TYPES = ["void"]

    def __init__(self, name: str, cuda_type: str, include_files: Optional[List[str]] = None):
        self.name = name
        self.cuda_type = cuda_type
        self.include_files = include_files

    def __str__(self):
        return self.cuda_type


# TODO: These need to be checked (include missing)
void = dtype("void", "void")
int1 = dtype("int1", "int1_t")
int8 = dtype("int8", "int8_t")
int16 = dtype("int16", "int16_t")
int32 = dtype("int32", "int32_t")
int64 = dtype("int64", "int64_t")
uint8 = dtype("uint8", "uint8_t")
uint16 = dtype("uint16", "uint16_t")
uint32 = dtype("uint32", "uint32_t")
uint64 = dtype("uint64", "uint64_t")
float16 = dtype("fp16", "half")
bfloat16 = dtype("bf16", "nv_bfloat16")
float32 = dtype("fp32", "float")
float64 = dtype("fp64", "double")

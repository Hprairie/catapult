class dtype:
    SINT_TYPES = ["int8", "int16", "int32", "int64"]
    UINT_TYPES = ["int1", "uint8", "uint16", "uint32", "uint64"]
    FP_TYPES = ["fp16", "bf16", "fp32", "fp64"]
    OTHER_TYPES = ["void"]

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name


void = dtype("void")
int1 = dtype("int1")
int8 = dtype("int8")
int16 = dtype("int16")
int32 = dtype("int32")
int64 = dtype("int64")
uint8 = dtype("uint8")
uint16 = dtype("uint16")
uint32 = dtype("uint32")
uint64 = dtype("uint64")
float16 = dtype("fp16")
bfloat16 = dtype("bf16")
float32 = dtype("fp32")
float64 = dtype("fp64")

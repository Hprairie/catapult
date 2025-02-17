import pytest
from unittest.mock import patch, MagicMock
from catapult.compiler import _NVRTCProgram, CompileException  # Direct import of the class


# Test data
DUMMY_KERNEL_PATH = "dummy_kernel.cu"
INVALID_KERNEL_PATH = "invalid_kernel.cu"


@pytest.fixture
def mock_nvrtc_create_program():
    with patch('catapult.nvrtc.nvrtcCreateProgram') as mock_create_program:
        yield mock_create_program


@pytest.fixture
def mock_nvrtc_compile_program():
    with patch('catapult.nvrtc.nvrtcCompileProgram') as mock_compile_program:
        yield mock_compile_program


@pytest.fixture
def mock_check_cuda_errors():
    with patch('catapult.checkCudaErrors') as mock_check_errors:
        yield mock_check_errors


def test_kernel_program_creation(mock_nvrtc_create_program, mock_check_cuda_errors):
    """Test the creation of a kernel program using the provided source."""
    
    source = "kernel code here"
    name = "test_kernel"
    
    # Mock the function call to avoid actual CUDA calls
    mock_nvrtc_create_program.return_value = MagicMock()  # Return a mock program object
    mock_check_cuda_errors.return_value = None  # Simulate no errors
    
    # Directly instantiate _NVRTCProgram
    program = _NVRTCProgram(
        source=source.encode("utf-8"),
        name=name.encode("utf-8"),
        num_headers=0,  # No headers
        headers=None,
        include_names=None
    )
    
    # Ensure that nvrtcCreateProgram was called with the correct arguments
    mock_nvrtc_create_program.assert_called_once_with(
        source.encode("utf-8"),
        name.encode("utf-8"),
        0,  # Assuming no headers are used
        None,
        None
    )
    assert isinstance(program, _NVRTCProgram)
    assert program.name == "test_kernel"
    assert program.source == source

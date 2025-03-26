import pytest
import torch

@pytest.fixture
def cuda_device():
    """Fixture to ensure CUDA is available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("Skipping test: CUDA device not available")
    return device

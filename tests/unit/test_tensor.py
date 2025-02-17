# import pytest
# import torch
# from catapult import KernelLauncher, ConfigManager

# # Sample dummy kernel path for testing
# DUMMY_KERNEL_PATH = "tests/dummy_kernel.cu"

# def test_tensor_shape_check():
#     """Test that tensors passed to the kernel launcher are validated."""
#     launcher = KernelLauncher(DUMMY_KERNEL_PATH)
#     tensor = torch.randn(10, 10)
#     assert launcher._validate_tensor_shape(tensor)  # Assuming this method exists

# def test_tensor_shape_check_invalid():
#     """Test that invalid tensor shapes raise an error."""
#     launcher = KernelLauncher(DUMMY_KERNEL_PATH)
#     invalid_tensor = torch.tensor([1, 2, 3])  # Assuming only 2D tensors are valid
#     with pytest.raises(ValueError):
#         launcher._validate_tensor_shape(invalid_tensor)
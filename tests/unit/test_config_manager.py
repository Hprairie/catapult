# import pytest
# import torch
# from catapult import KernelLauncher, ConfigManager

# # Sample dummy kernel path for testing
# DUMMY_KERNEL_PATH = "tests/dummy_kernel.cu"

# def test_config_manager_add_config():
#     """Test adding a configuration to the ConfigManager."""
#     config_manager = ConfigManager()
#     config = {"block_size": 256, "grid_size": 16}
#     config_manager.add_config("test_config", config)
#     assert "test_config" in config_manager.configs
#     assert config_manager.configs["test_config"] == config

# def test_config_manager_invalid_config():
#     """Test that adding an invalid config format raises an error."""
#     config_manager = ConfigManager()
#     with pytest.raises(TypeError):
#         config_manager.add_config("invalid_config", "not_a_dict")

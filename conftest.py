def pytest_addoption(parser) -> None:
    parser.addoption("--layer-artifact", default=None, help="Real layer14.safetensors from a run")
    parser.addoption("--calibration-artifact", default=None, help="Real calibration.safetensors from the same run")

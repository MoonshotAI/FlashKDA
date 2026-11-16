def pytest_addoption(parser):
    parser.addoption("--run-long", action="store_true", default=False,
                     help="run 128K/1M-token NPU correctness cases")

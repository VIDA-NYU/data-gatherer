import os
import pytest
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)
@pytest.fixture
def get_test_data_path():
    def _get_path(filename):
        return os.path.join(os.path.dirname(__file__), 'test_data', filename)
    return _get_path
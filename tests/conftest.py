# conftest.py
import pytest
import logging
import os
from datetime import datetime

log_directory = "test_logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_filename = f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(log_directory, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

@pytest.fixture(autouse=True)
def log_test_info(request):
    logging.info(f"Starting test: {request.node.name}")
    yield
    logging.info(f"Finished test: {request.node.name}")

@pytest.fixture
def mock_video_capture(mocker):
    mock_cap = mocker.Mock()
    mock_cap.read.return_value = (True, None)
    mock_cap.release = mocker.Mock()
    return mock_cap

@pytest.fixture
def mock_cv2(mocker):
    return mocker.patch('cv2')
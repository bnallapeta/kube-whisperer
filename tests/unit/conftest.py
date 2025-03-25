import pytest
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import ServiceConfig, ModelConfig, WhisperModel, DeviceType, ComputeType
from src.utils import AudioValidator, TempFileManager

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration."""
    return ServiceConfig(
        model=ModelConfig(
            whisper_model=WhisperModel.TINY,
            cpu_threads=2,
            num_workers=1
        ),
        temp_dir=temp_dir,
        max_file_size=1024 * 1024,  # 1MB
        batch_size=2,
        request_timeout=30
    )

@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a sample audio file for testing."""
    audio_file = os.path.join(temp_dir, "test.wav")
    # Create a minimal WAV file header
    with open(audio_file, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write((36).to_bytes(4, 'little'))  # File size
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write((16).to_bytes(4, 'little'))  # Chunk size
        f.write((1).to_bytes(2, 'little'))   # Audio format (PCM)
        f.write((1).to_bytes(2, 'little'))   # Num channels
        f.write((44100).to_bytes(4, 'little'))  # Sample rate
        f.write((88200).to_bytes(4, 'little'))  # Byte rate
        f.write((2).to_bytes(2, 'little'))   # Block align
        f.write((16).to_bytes(2, 'little'))  # Bits per sample
        # data chunk
        f.write(b"data")
        f.write((0).to_bytes(4, 'little'))   # Data size
    yield audio_file
    os.remove(audio_file)

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set mock environment variables."""
    env_vars = {
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "DEBUG",
        "JSON_LOGS": "true",
        "BATCH_SIZE": "4",
        "REQUEST_TIMEOUT": "60"
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars

@pytest.fixture
def service_config():
    """Fixture for service configuration"""
    return ServiceConfig(
        host="localhost",
        port=8000,
        log_level="INFO",
        environment="test",
        max_file_size_mb=25,
        allowed_file_types=["audio/wav", "audio/mp3", "audio/mpeg"],
        temp_dir="/tmp/whisper_test",
        cleanup_interval_minutes=60
    )

@pytest.fixture
def model_config():
    """Fixture for model configuration"""
    return ModelConfig(
        whisper_model=WhisperModel.TINY,
        device=DeviceType.CPU,
        compute_type=ComputeType.FLOAT32,
        download_root="/tmp/whisper_models_test",
        language="en",
        cpu_threads=4,
        num_workers=2
    )

@pytest.fixture
def temp_file_manager(service_config):
    """Fixture for temporary file manager"""
    manager = TempFileManager(
        temp_dir=service_config.temp_dir,
        cleanup_interval_minutes=service_config.cleanup_interval_minutes
    )
    yield manager
    # Cleanup after tests
    manager.cleanup_temp_files()

@pytest.fixture
def audio_validator(service_config):
    """Fixture for audio file validator"""
    return AudioValidator(
        max_file_size_mb=service_config.max_file_size_mb,
        allowed_file_types=service_config.allowed_file_types
    ) 
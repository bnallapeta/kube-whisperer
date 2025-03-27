import pytest
from src.config import (
    WhisperModel,
    ComputeType,
    DeviceType,
    AudioFormat,
    TranscriptionOptions,
    ModelConfig,
    ServiceConfig
)
import os
from pydantic import ValidationError
import asyncio
from unittest.mock import Mock

def test_whisper_model_enum():
    """Test WhisperModel enum values."""
    assert WhisperModel.TINY.value == "tiny"
    assert WhisperModel.BASE.value == "base"
    assert WhisperModel.SMALL.value == "small"
    assert WhisperModel.MEDIUM.value == "medium"
    assert WhisperModel.LARGE.value == "large"

def test_compute_type_enum():
    """Test ComputeType enum values."""
    assert ComputeType.FLOAT32.value == "float32"
    assert ComputeType.FLOAT16.value == "float16"
    assert ComputeType.INT8.value == "int8"

def test_device_type_enum():
    """Test DeviceType enum values."""
    assert DeviceType.AUTO.value == "auto"
    assert DeviceType.CPU.value == "cpu"
    assert DeviceType.CUDA.value == "cuda"

def test_audio_format_enum():
    """Test AudioFormat enum values."""
    assert AudioFormat.WAV.value == "audio/wav"
    assert AudioFormat.MP3.value == "audio/mp3"
    assert AudioFormat.FLAC.value == "audio/flac"

def test_transcription_options_validation():
    """Test TranscriptionOptions validation."""
    # Test valid config
    config = TranscriptionOptions(
        language="en",
        task="transcribe",
        beam_size=5,
        patience=1.0,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    assert config.language == "en"
    assert config.task == "transcribe"
    assert config.beam_size == 5
    assert config.patience == 1.0
    assert len(config.temperature) == 6

    # Test invalid language code
    with pytest.raises(ValidationError):
        TranscriptionOptions(language="eng")  # Must be 2 letters

    # Test invalid task
    with pytest.raises(ValidationError):
        TranscriptionOptions(task="invalid")  # Must be transcribe or translate

    # Test invalid beam size
    with pytest.raises(ValidationError):
        TranscriptionOptions(beam_size=0)  # Must be between 1 and 10

    # Test invalid patience
    with pytest.raises(ValidationError):
        TranscriptionOptions(patience=2.5)  # Must be between 0.0 and 2.0

    # Test invalid temperature
    with pytest.raises(ValidationError):
        TranscriptionOptions(temperature=[1.5])  # Must be between 0.0 and 1.0

    # Test default values
    default_config = TranscriptionOptions()
    assert default_config.language == "en"  # Now defaults to "en"
    assert default_config.task == "transcribe"
    assert default_config.beam_size == 5
    assert default_config.patience == 1.0
    assert default_config.temperature == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

def test_model_config_validation():
    """Test ModelConfig validation."""
    # Test valid config
    config = ModelConfig(
        whisper_model=WhisperModel.BASE,
        device=DeviceType.CPU,
        compute_type=ComputeType.FLOAT32,
        cpu_threads=4,
        num_workers=2,
        download_root="/tmp/whisper_models",
        language="en"
    )
    assert config.whisper_model == WhisperModel.BASE
    assert config.device == DeviceType.CPU
    assert config.compute_type == ComputeType.FLOAT32
    assert config.cpu_threads == 4
    assert config.num_workers == 2
    assert config.download_root == "/tmp/whisper_models"
    assert config.language == "en"

    # Test auto device selection
    auto_config = ModelConfig(device=DeviceType.AUTO)
    assert auto_config.device == DeviceType.AUTO

    # Test invalid compute type
    with pytest.raises(ValidationError):
        ModelConfig(compute_type="invalid_type")

    # Test invalid CPU threads
    with pytest.raises(ValidationError):
        ModelConfig(cpu_threads=-1)

    # Test invalid num_workers
    with pytest.raises(ValidationError):
        ModelConfig(num_workers=0)

    # Test invalid language
    with pytest.raises(ValidationError):
        ModelConfig(language="invalid")

    # Test default values
    default_config = ModelConfig()
    assert default_config.whisper_model == WhisperModel.BASE
    assert default_config.device == DeviceType.AUTO
    assert default_config.compute_type == ComputeType.FLOAT32
    assert default_config.cpu_threads == 4
    assert default_config.num_workers == 2
    assert default_config.language == "en"

def test_health_status_values():
    """Test health status values."""
    from src.serve import HealthStatus

    # Create a mock model and config
    mock_model = Mock()
    mock_config = ModelConfig()
    health_checker = HealthStatus(mock_model, mock_config)

    # Test model health check
    result = asyncio.run(health_checker.check_model_health())
    assert result["status"] in ["healthy", "error"]

    # Test GPU health check
    result = asyncio.run(health_checker.check_gpu_health())
    assert result["status"] in ["healthy", "not_available", "error"]

    # Test system health check
    result = asyncio.run(health_checker.check_system_health())
    assert result["status"] in ["healthy", "warning", "error"]

    # Test temp directory check
    result = asyncio.run(health_checker.check_temp_directory())
    assert result["status"] in ["healthy", "warning", "error"]

def test_service_config_env_vars(monkeypatch):
    """Test ServiceConfig with environment variables."""
    # Set environment variables
    test_temp_dir = os.path.join(os.path.expanduser("~"), "test_whisper_temp")
    env_vars = {
        "WHISPER_ENVIRONMENT": "production",
        "WHISPER_LOG_LEVEL": "DEBUG",
        "WHISPER_JSON_LOGS": "false",
        "WHISPER_TEMP_DIR": test_temp_dir,
        "WHISPER_BATCH_SIZE": "16",
        "WHISPER_REQUEST_TIMEOUT": "600"
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    try:
        # Create config with environment variables
        config = ServiceConfig()
        
        # Test environment variable overrides
        assert config.environment == "production"
        assert config.log_level == "DEBUG"
        assert config.json_logs is False
        assert config.temp_dir == test_temp_dir
        assert config.batch_size == 16
        assert config.request_timeout == 600
    finally:
        # Clean up test directory
        if os.path.exists(test_temp_dir):
            os.rmdir(test_temp_dir)

def test_service_config_methods():
    """Test ServiceConfig helper methods."""
    config = ServiceConfig()
    
    # Test get_model_config
    model_config = config.get_model_config()
    assert isinstance(model_config, ModelConfig)
    assert model_config.whisper_model == WhisperModel.BASE
    
    # Test get_allowed_formats
    allowed_formats = config.get_allowed_formats()
    assert isinstance(allowed_formats, list)
    assert "audio/wav" in allowed_formats
    assert "audio/mp3" in allowed_formats
    
    # Test get_temp_dir
    temp_dir = config.get_temp_dir()
    assert os.path.exists(temp_dir)
    assert os.path.isdir(temp_dir)

@pytest.mark.config
def test_service_config_validation(service_config):
    """Test service configuration validation"""
    assert service_config.host == "localhost"
    assert service_config.port == 8000
    assert service_config.log_level == "INFO"
    assert service_config.environment == "test"
    assert service_config.max_file_size_mb == 25
    assert service_config.allowed_file_types == ["audio/wav", "audio/mp3", "audio/mpeg"]
    assert service_config.temp_dir == "/tmp/whisper_test"
    assert service_config.cleanup_interval_minutes == 60

@pytest.mark.config
def test_model_config_validation(model_config):
    """Test model configuration validation"""
    assert model_config.whisper_model == WhisperModel.TINY
    assert model_config.device == DeviceType.CPU
    assert model_config.compute_type == ComputeType.FLOAT32
    assert model_config.download_root == "/tmp/whisper_models_test"
    assert model_config.language == "en"
    assert model_config.cpu_threads == 4
    assert model_config.num_workers == 2

@pytest.mark.config
def test_invalid_service_config():
    """Test invalid service configuration"""
    with pytest.raises(ValueError):
        ServiceConfig(
            host="localhost",
            port=-1,  # Invalid port
            log_level="INFO",
            environment="test",
            max_file_size_mb=25,
            allowed_file_types=["audio/wav"],
            temp_dir="/tmp/whisper_test",
            cleanup_interval_minutes=60
        )

    with pytest.raises(ValueError):
        ServiceConfig(
            host="localhost",
            port=8000,
            log_level="INVALID",  # Invalid log level
            environment="test",
            max_file_size_mb=25,
            allowed_file_types=["audio/wav"],
            temp_dir="/tmp/whisper_test",
            cleanup_interval_minutes=60
        )

    with pytest.raises(ValueError):
        ServiceConfig(
            host="localhost",
            port=8000,
            log_level="INFO",
            environment="invalid",  # Invalid environment
            max_file_size_mb=25,
            allowed_file_types=["audio/wav"],
            temp_dir="/tmp/whisper_test",
            cleanup_interval_minutes=60
        )

@pytest.mark.config
def test_invalid_model_config():
    """Test invalid model configuration"""
    with pytest.raises(ValueError):
        ModelConfig(
            whisper_model="invalid_model",  # Invalid model type
            device=DeviceType.CPU,
            compute_type=ComputeType.FLOAT32,
            download_root="/tmp/whisper_models_test",
            language="en",
            cpu_threads=4,
            num_workers=2
        )

    with pytest.raises(ValueError):
        ModelConfig(
            whisper_model=WhisperModel.TINY,
            device=DeviceType.CPU,
            compute_type=ComputeType.FLOAT32,
            download_root="/tmp/whisper_models_test",
            language="x",  # Invalid language code
            cpu_threads=4,
            num_workers=2
        )

@pytest.mark.config
def test_environment_specific_config():
    """Test environment-specific configuration"""
    # Test production settings
    prod_config = ServiceConfig(
        host="0.0.0.0",
        port=8000,
        log_level="WARNING",
        environment="production",
        max_file_size_mb=100,
        allowed_file_types=["audio/wav", "audio/mp3", "audio/mpeg"],
        temp_dir="/tmp/whisper",
        cleanup_interval_minutes=30
    )
    assert prod_config.environment == "production"
    assert prod_config.log_level == "WARNING"
    assert prod_config.cleanup_interval_minutes == 30

    # Test development settings
    dev_config = ServiceConfig(
        host="localhost",
        port=8000,
        log_level="DEBUG",
        environment="development",
        max_file_size_mb=50,
        allowed_file_types=["audio/wav", "audio/mp3", "audio/mpeg"],
        temp_dir="/tmp/whisper_dev",
        cleanup_interval_minutes=120
    )
    assert dev_config.environment == "development"
    assert dev_config.log_level == "DEBUG"
    assert dev_config.cleanup_interval_minutes == 120 
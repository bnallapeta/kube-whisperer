import pytest
from fastapi.testclient import TestClient
import torch
import os
import json
import whisper
from unittest.mock import Mock, patch, MagicMock
from src.serve import (
    app, load_model, HealthStatus, ModelConfig,
    TranscriptionOptions, BatchRequest, TEMP_DIR
)

@pytest.fixture
def test_client():
    """Create a test client."""
    return TestClient(app)

@pytest.fixture
def mock_model():
    """Create a mock Whisper model."""
    mock = Mock(spec=whisper.Whisper)
    mock.device = torch.device("cpu")
    mock.parameters = MagicMock(return_value=[torch.tensor([1.0])])
    return mock

@pytest.fixture
def mock_config():
    """Create a mock model configuration."""
    return ModelConfig(
        whisper_model="base",
        device="cpu",
        compute_type="float32",
        cpu_threads=4,
        num_workers=2
    )

@pytest.fixture
def health_status(mock_model, mock_config):
    """Create a HealthStatus instance with mock model."""
    return HealthStatus(mock_model, mock_config)

def test_load_model_cpu():
    """Test loading model on CPU."""
    config = ModelConfig(device="cpu")
    with patch("whisper.load_model") as mock_load:
        mock_model = Mock()
        mock_load.return_value = mock_model
        model = load_model(config)
        assert model == mock_model
        mock_load.assert_called_once_with("base", device="cpu")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_model_gpu():
    """Test loading model on GPU."""
    config = ModelConfig(device="cuda", compute_type="float16")
    with patch("whisper.load_model") as mock_load:
        mock_model = Mock()
        mock_load.return_value = mock_model
        model = load_model(config)
        assert model == mock_model
        assert torch.get_default_dtype() == torch.float16

def test_load_model_auto():
    """Test auto device selection."""
    config = ModelConfig(device="auto")
    with patch("whisper.load_model") as mock_load:
        mock_model = Mock()
        mock_load.return_value = mock_model
        model = load_model(config)
        assert model == mock_model
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        mock_load.assert_called_once_with("base", device=expected_device)

@pytest.mark.asyncio
async def test_health_status_model_check(health_status):
    """Test model health check."""
    result = await health_status.check_model_health()
    assert result["status"] == "healthy"
    assert result["device"] == "cpu"
    assert result["model_name"] == "base"
    assert result["compute_type"] == "float32"

@pytest.mark.asyncio
async def test_health_status_gpu_check(health_status):
    """Test GPU health check."""
    result = await health_status.check_gpu_health()
    if torch.cuda.is_available():
        assert result["status"] == "healthy"
        assert "device_count" in result
        assert "device_name" in result
    else:
        assert result["status"] == "not_available"

@pytest.mark.asyncio
async def test_health_status_system_check(health_status):
    """Test system health check."""
    result = await health_status.check_system_health()
    assert result["status"] == "healthy"
    assert "cpu_percent" in result
    assert "memory_percent" in result
    assert "thread_count" in result

@pytest.mark.asyncio
async def test_health_status_temp_directory(health_status):
    """Test temporary directory check."""
    result = await health_status.check_temp_directory()
    assert result["status"] == "healthy"
    assert result["path"] == TEMP_DIR
    assert "free_space" in result
    assert "total_space" in result

def test_transcription_options_validation():
    """Test transcription options validation."""
    # Valid options
    options = TranscriptionOptions(
        language="en",
        task="transcribe",
        beam_size=5,
        patience=1.0,
        temperature=[0.0, 0.2, 0.4]
    )
    assert options.language == "en"
    assert options.task == "transcribe"
    assert options.beam_size == 5

    # Invalid task
    with pytest.raises(ValueError):
        TranscriptionOptions(task="invalid_task")

def test_batch_request_validation():
    """Test batch request validation."""
    # Valid request
    request = BatchRequest(
        files=["file1.wav", "file2.wav"],
        options=TranscriptionOptions(language="en")
    )
    assert len(request.files) == 2
    assert request.options.language == "en"

    # Empty files list
    with pytest.raises(ValueError):
        BatchRequest(files=[])

def test_metrics_endpoint(test_client):
    """Test metrics endpoint."""
    response = test_client.get("/metrics")
    assert response.status_code == 200
    assert "whisper_requests_total" in response.text
    assert "whisper_errors_total" in response.text
    assert "whisper_processing_seconds" in response.text

def test_cors_headers(test_client):
    """Test CORS headers."""
    response = test_client.options("/")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert response.headers["access-control-allow-methods"]
    assert response.headers["access-control-allow-headers"]

@pytest.mark.asyncio
async def test_health_check_endpoint(test_client):
    """Test health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "gpu" in data
    assert "system" in data
    assert "temp_directory" in data

@pytest.mark.asyncio
async def test_readiness_check_endpoint(test_client):
    """Test readiness check endpoint."""
    response = test_client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ready", "not_ready"]

@pytest.mark.asyncio
async def test_liveness_check_endpoint(test_client):
    """Test liveness check endpoint."""
    response = test_client.get("/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"

def test_error_handling(test_client):
    """Test error handling middleware."""
    with patch("src.serve.load_model", side_effect=RuntimeError("Test error")):
        response = test_client.post("/transcribe", files={"file": ("test.wav", b"invalid audio")})
        assert response.status_code == 500
        assert "error" in response.json()

def test_temp_directory_creation():
    """Test temporary directory creation."""
    assert os.path.exists(TEMP_DIR)
    assert os.path.isdir(TEMP_DIR)

def test_model_config_validation():
    """Test model configuration validation."""
    # Valid config
    config = ModelConfig(
        whisper_model="base",
        device="cpu",
        compute_type="float32",
        cpu_threads=4,
        num_workers=2
    )
    assert config.whisper_model == "base"
    assert config.device == "cpu"

    # Invalid compute type
    with pytest.raises(ValueError):
        ModelConfig(compute_type="invalid_type")

    # Invalid CPU threads
    with pytest.raises(ValueError):
        ModelConfig(cpu_threads=-1)

@pytest.mark.asyncio
async def test_transcribe_endpoint(test_client):
    """Test transcribe endpoint."""
    # Create a mock audio file
    audio_content = b"mock audio content"
    files = {"file": ("test.wav", audio_content)}
    
    with patch("src.serve.validate_audio_file") as mock_validate:
        mock_validate.return_value = {"duration": 10, "sample_rate": 16000}
        
        response = test_client.post(
            "/transcribe",
            files=files,
            data={"language": "en"}
        )
        
        assert response.status_code in [200, 202]  # 202 if async processing
        data = response.json()
        assert "task_id" in data or "text" in data

@pytest.mark.asyncio
async def test_batch_transcribe_endpoint(test_client):
    """Test batch transcription endpoint."""
    request_data = {
        "files": ["file1.wav", "file2.wav"],
        "options": {
            "language": "en",
            "task": "transcribe"
        }
    }
    
    with patch("src.serve.validate_audio_file") as mock_validate:
        mock_validate.return_value = {"duration": 10, "sample_rate": 16000}
        
        response = test_client.post(
            "/batch_transcribe",
            json=request_data
        )
        
        assert response.status_code in [200, 202]
        data = response.json()
        assert "task_ids" in data or "results" in data 
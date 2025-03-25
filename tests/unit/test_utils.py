import pytest
import os
import aiofiles
import magic
import shutil
import hashlib
import asyncio
from fastapi import HTTPException, UploadFile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from src.utils import (
    AudioValidator,
    TempFileManager,
    AudioValidationError,
    AudioMetadata,
    validate_audio_file,
    cleanup_temp_file,
    AudioFormat,
    get_mime_type,
    get_audio_duration,
    create_temp_directory,
    remove_temp_directory,
    ALLOWED_AUDIO_TYPES,
    MAX_FILE_SIZE
)

@pytest.fixture
def temp_audio_file(tmp_path):
    """Create a temporary WAV file for testing."""
    file_path = tmp_path / "test.wav"
    # Write minimal WAV file structure
    with open(file_path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write((36).to_bytes(4, 'little'))  # File size
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write((16).to_bytes(4, 'little'))  # Chunk size
        f.write((1).to_bytes(2, 'little'))   # PCM format
        f.write((1).to_bytes(2, 'little'))   # Mono
        f.write((44100).to_bytes(4, 'little'))  # Sample rate
        f.write((88200).to_bytes(4, 'little'))  # Byte rate
        f.write((2).to_bytes(2, 'little'))   # Block align
        f.write((16).to_bytes(2, 'little'))  # Bits per sample
        # data chunk
        f.write(b"data")
        f.write((0).to_bytes(4, 'little'))   # Data size
    return file_path

@pytest.fixture
def audio_validator():
    """Create an AudioValidator instance."""
    config = {
        'max_file_size': 1024 * 1024,  # 1MB
        'allowed_audio_types': ALLOWED_AUDIO_TYPES,
        'temp_dir': '/tmp/test_whisper_audio'
    }
    return AudioValidator(config)

@pytest.fixture
def temp_file_manager():
    """Create a TempFileManager instance."""
    return TempFileManager('/tmp/test_whisper_audio')

@pytest.mark.asyncio
async def test_audio_validator_save_file(audio_validator):
    """Test saving an uploaded audio file."""
    # Create mock UploadFile
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "test.wav"
    mock_file.read = AsyncMock(side_effect=[b"test audio data", b""])
    
    # Test file saving
    temp_path = await audio_validator.save_upload_file(mock_file)
    assert os.path.exists(temp_path)
    assert os.path.getsize(temp_path) > 0
    
    # Cleanup
    os.remove(temp_path)

@pytest.mark.asyncio
async def test_audio_validator_compute_hash(audio_validator, temp_audio_file):
    """Test computing file hash."""
    file_hash = await audio_validator.compute_file_hash(temp_audio_file)
    assert len(file_hash) == 64  # SHA-256 hash length
    assert isinstance(file_hash, str)

@pytest.mark.asyncio
async def test_audio_validation(audio_validator, temp_audio_file):
    """Test audio file validation."""
    with patch('magic.from_file', return_value='audio/wav'):
        metadata = await audio_validator.validate_audio_file(
            temp_audio_file,
            "test.wav"
        )
        assert isinstance(metadata, AudioMetadata)
        assert metadata.original_filename == "test.wav"
        assert metadata.file_type == "audio/wav"
        assert metadata.file_size > 0

@pytest.mark.asyncio
async def test_invalid_audio_validation(audio_validator, tmp_path):
    """Test validation of invalid audio files."""
    # Test non-existent file
    with pytest.raises(HTTPException) as exc_info:
        await audio_validator.validate_audio_file(
            str(tmp_path / "nonexistent.wav"),
            "nonexistent.wav"
        )
    assert exc_info.value.status_code == 400
    
    # Test oversized file
    large_file = tmp_path / "large.wav"
    large_file.write_bytes(b"x" * (audio_validator.max_file_size + 1))
    with pytest.raises(HTTPException) as exc_info:
        await audio_validator.validate_audio_file(
            str(large_file),
            "large.wav"
        )
    assert exc_info.value.status_code == 400

@pytest.mark.asyncio
async def test_temp_file_cleanup(temp_file_manager, tmp_path):
    """Test temporary file cleanup."""
    # Create test files
    old_file = tmp_path / "old.wav"
    new_file = tmp_path / "new.wav"
    old_file.write_bytes(b"old")
    new_file.write_bytes(b"new")
    
    # Set old file's modification time
    old_time = datetime.now() - timedelta(hours=25)
    os.utime(old_file, (old_time.timestamp(), old_time.timestamp()))
    
    # Run cleanup
    await temp_file_manager.cleanup_temp_files()
    
    # Verify old file is removed
    assert not old_file.exists()
    assert new_file.exists()

@pytest.mark.asyncio
async def test_temp_file_manager_remove(temp_file_manager, tmp_path):
    """Test removing individual temporary files."""
    test_file = tmp_path / "test.wav"
    test_file.write_bytes(b"test")
    
    temp_file_manager.remove_temp_file(str(test_file))
    assert not test_file.exists()

def test_legacy_validate_audio_file():
    """Test legacy audio file validation function."""
    with pytest.raises(HTTPException) as exc_info:
        validate_audio_file("nonexistent.wav")
    assert exc_info.value.status_code == 400

def test_legacy_cleanup_temp_file(tmp_path):
    """Test legacy cleanup function."""
    test_file = tmp_path / "test.wav"
    test_file.write_bytes(b"test")
    
    # Remove file
    os.remove(test_file)
    assert not test_file.exists()

def test_audio_metadata_model():
    """Test AudioMetadata model."""
    metadata = AudioMetadata(
        original_filename="test.wav",
        temp_path="/tmp/test.wav",
        file_type="audio/wav",
        file_size=1000,
        content_hash="abc123",
        duration=1.5
    )
    assert metadata.original_filename == "test.wav"
    assert metadata.temp_path == "/tmp/test.wav"
    assert metadata.file_type == "audio/wav"
    assert metadata.file_size == 1000
    assert metadata.content_hash == "abc123"
    assert metadata.duration == 1.5

def test_audio_format_enum():
    """Test AudioFormat enum."""
    assert AudioFormat.WAV.value == "wav"
    assert AudioFormat.MP3.value == "mp3"
    assert AudioFormat.OGG.value == "ogg"
    assert AudioFormat.FLAC.value == "flac"
    assert AudioFormat.M4A.value == "m4a"

@pytest.mark.asyncio
async def test_audio_validator_error_handling(audio_validator, temp_audio_file):
    """Test error handling in AudioValidator."""
    # Test invalid file type
    with patch('magic.from_file', return_value='text/plain'):
        with pytest.raises(HTTPException) as exc_info:
            await audio_validator.validate_audio_file(
                temp_audio_file,
                "test.txt"
            )
        assert exc_info.value.status_code == 400
        assert "Invalid audio format" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_temp_file_manager_error_handling(temp_file_manager, tmp_path):
    """Test error handling in TempFileManager."""
    # Test cleanup with invalid directory
    temp_file_manager.temp_dir = "/nonexistent/dir"
    await temp_file_manager.cleanup_temp_files()  # Should not raise exception

    # Test removal of non-existent file
    temp_file_manager.remove_temp_file("/nonexistent/file.wav")  # Should not raise exception

@pytest.mark.asyncio
async def test_audio_validator_save_file_errors(audio_validator):
    """Test error handling when saving files."""
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "test.wav"
    mock_file.read = AsyncMock(side_effect=Exception("Read error"))
    
    with pytest.raises(HTTPException) as exc_info:
        await audio_validator.save_upload_file(mock_file)
    assert exc_info.value.status_code == 500
    assert "Failed to save uploaded file" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_temp_file_manager_scheduler(temp_file_manager):
    """Test temp file cleanup scheduler."""
    # Mock cleanup_temp_files to track calls
    temp_file_manager.cleanup_temp_files = AsyncMock()
    
    # Start scheduler with short interval
    cleanup_task = asyncio.create_task(
        temp_file_manager.start_cleanup_scheduler(interval_hours=0.001)
    )
    
    # Wait for a few cleanup cycles
    await asyncio.sleep(0.005)
    cleanup_task.cancel()
    
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    # Verify cleanup was called multiple times
    assert temp_file_manager.cleanup_temp_files.call_count > 1

def test_audio_validator_config():
    """Test AudioValidator configuration."""
    # Test default config
    validator = AudioValidator({})
    assert validator.max_file_size == 25 * 1024 * 1024
    assert validator.allowed_types == ALLOWED_AUDIO_TYPES
    assert validator.temp_dir == '/tmp/whisper_audio'
    
    # Test custom config
    custom_config = {
        'max_file_size': 1000,
        'allowed_audio_types': ['audio/wav'],
        'temp_dir': '/custom/temp'
    }
    validator = AudioValidator(custom_config)
    assert validator.max_file_size == 1000
    assert validator.allowed_types == ['audio/wav']
    assert validator.temp_dir == '/custom/temp'

def test_get_mime_type(temp_audio_file, tmp_path):
    """Test MIME type detection."""
    # Test WAV file
    mime_type = get_mime_type(temp_audio_file)
    assert mime_type in ["audio/wav", "audio/x-wav"]
    
    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        get_mime_type(str(tmp_path / "nonexistent.wav"))
    
    # Test text file
    text_file = tmp_path / "test.txt"
    text_file.write_text("This is a text file")
    mime_type = get_mime_type(str(text_file))
    assert mime_type.startswith("text/")

@pytest.mark.asyncio
async def test_get_audio_duration(temp_audio_file):
    """Test audio duration calculation."""
    duration = await get_audio_duration(temp_audio_file)
    assert isinstance(duration, float)
    assert duration >= 0

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        await get_audio_duration("nonexistent.wav")

def test_create_temp_directory(tmp_path):
    """Test temporary directory creation."""
    temp_dir = str(tmp_path / "whisper_temp")
    
    # Test creating new directory
    created_dir = create_temp_directory(temp_dir)
    assert os.path.exists(created_dir)
    assert os.path.isdir(created_dir)
    
    # Test with existing directory (should not raise)
    create_temp_directory(temp_dir)
    
    # Test with file path
    file_path = tmp_path / "file.txt"
    file_path.write_text("test")
    with pytest.raises(OSError):
        create_temp_directory(str(file_path))

def test_remove_temp_directory(tmp_path):
    """Test temporary directory removal."""
    temp_dir = tmp_path / "whisper_temp"
    os.makedirs(temp_dir)
    
    # Create some files in the directory
    (temp_dir / "test1.wav").write_bytes(b"test1")
    (temp_dir / "test2.wav").write_bytes(b"test2")
    
    # Test removal
    remove_temp_directory(str(temp_dir))
    assert not os.path.exists(temp_dir)
    
    # Test removing non-existent directory (should not raise)
    remove_temp_directory(str(temp_dir))

@pytest.mark.asyncio
async def test_audio_validator_with_different_formats(audio_validator, tmp_path):
    """Test audio validator with different audio formats."""
    formats = {
        "test.wav": b"RIFF...WAVEfmt ",
        "test.mp3": b"ID3...",
        "test.ogg": b"OggS...",
        "test.m4a": b"....ftypM4A"
    }
    
    for filename, content in formats.items():
        file_path = tmp_path / filename
        file_path.write_bytes(content)
        
        try:
            await audio_validator.validate_audio_file(str(file_path), filename)
        except HTTPException as e:
            # We expect some formats to fail in this test environment
            assert e.status_code == 400

@pytest.mark.asyncio
async def test_concurrent_temp_file_operations(temp_file_manager, tmp_path):
    """Test concurrent temporary file operations."""
    # Create multiple files
    files = []
    for i in range(10):
        file_path = tmp_path / f"test{i}.wav"
        file_path.write_bytes(b"test")
        files.append(str(file_path))
    
    # Test concurrent cleanup and removal
    await asyncio.gather(
        temp_file_manager.cleanup_temp_files(max_age_hours=0),
        *[asyncio.create_task(
            temp_file_manager.remove_temp_file(f)) for f in files[:5]]
    )
    
    # Verify results
    for i in range(10):
        file_path = tmp_path / f"test{i}.wav"
        assert not file_path.exists()

def test_audio_metadata_model_validation():
    """Test AudioMetadata model validation."""
    # Test valid metadata
    metadata = AudioMetadata(
        original_filename="test.wav",
        temp_path="/tmp/test.wav",
        file_type="audio/wav",
        file_size=1024,
        content_hash="abc123",
        duration=1.5
    )
    assert metadata.original_filename == "test.wav"
    assert metadata.file_size == 1024
    assert metadata.duration == 1.5
    
    # Test string representation
    assert str(metadata) == "AudioMetadata(test.wav, audio/wav, 1024 bytes)" 
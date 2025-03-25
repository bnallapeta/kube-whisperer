from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from enum import Enum
from pathlib import Path

class WhisperModel(str, Enum):
    """Available Whisper model types."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

class DeviceType(str, Enum):
    """Available device types for model inference."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"

class ComputeType(str, Enum):
    """Available compute types for model inference."""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT8 = "int8"

class AudioFormat(str, Enum):
    """Supported audio formats."""
    WAV = "audio/wav"
    WAV_X = "audio/x-wav"
    MP3 = "audio/mp3"
    MPEG = "audio/mpeg"
    OGG = "audio/ogg"
    M4A = "audio/x-m4a"
    AAC = "audio/aac"
    FLAC = "audio/flac"

class TranscriptionOptions(BaseModel):
    """Configuration for transcription requests."""
    language: Optional[str] = Field(default=None, description="Language code (ISO 639-1)")
    task: str = Field(default="transcribe", description="Task type (transcribe or translate)")
    beam_size: int = Field(default=5, ge=1, le=10, description="Beam size for decoding")
    patience: float = Field(default=1.0, ge=0.0, le=2.0, description="Beam search patience factor")
    temperature: List[float] = Field(
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        description="Temperature for sampling"
    )
    initial_prompt: Optional[str] = Field(None, description="Optional text to provide as initial prompt")
    word_timestamps: bool = Field(False, description="Whether to include word-level timestamps")

    @field_validator("task")
    @classmethod
    def validate_task(cls, v):
        """Validate task type."""
        if v not in ["transcribe", "translate"]:
            raise ValueError("Task must be either 'transcribe' or 'translate'")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        """Validate language code."""
        if v and len(v) != 2:
            raise ValueError("Language code must be a 2-letter ISO 639-1 code")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature values."""
        if not all(0.0 <= t <= 1.0 for t in v):
            raise ValueError("Temperature values must be between 0.0 and 1.0")
        return v

class ModelConfig(BaseModel):
    """Model configuration settings."""
    whisper_model: WhisperModel = Field(
        default=WhisperModel.BASE,
        description="Whisper model type"
    )
    device: DeviceType = Field(
        default=DeviceType.CPU,
        description="Device type for model inference"
    )
    compute_type: ComputeType = Field(
        default=ComputeType.FLOAT32,
        description="Compute type for model inference"
    )
    language: str = Field(
        default="en",
        description="Default language for transcription"
    )
    download_root: str = Field(
        default="/tmp/whisper_models",
        description="Directory to store downloaded models"
    )
    cpu_threads: int = Field(
        default=4,
        description="Number of CPU threads to use",
        gt=0
    )
    num_workers: int = Field(
        default=2,
        description="Number of workers for parallel processing",
        gt=0
    )

    @field_validator("whisper_model", mode="before")
    def validate_whisper_model(cls, v: Union[str, WhisperModel]) -> WhisperModel:
        """Validate whisper model type."""
        if isinstance(v, str):
            try:
                return WhisperModel[v.upper()]
            except (KeyError, ValueError):
                raise ValueError(f"Invalid whisper model: {v}. Must be one of {list(WhisperModel.__members__.keys())}")
        elif isinstance(v, WhisperModel):
            return v
        raise ValueError(f"Invalid whisper model type: {type(v)}. Must be string or WhisperModel enum")

    @field_validator("language")
    def validate_language(cls, v: str) -> str:
        """Validate language code."""
        if not isinstance(v, str) or len(v) != 2:
            raise ValueError("Language code must be a 2-letter ISO code")
        return v.lower()

    @field_validator("download_root")
    def validate_download_root(cls, v: str) -> str:
        """Validate download root directory."""
        if not os.path.isabs(v):
            v = os.path.abspath(v)
        os.makedirs(v, exist_ok=True)
        return v

class ServiceConfig(BaseSettings):
    """Service configuration settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        env_prefix="WHISPER_"
    )

    host: str = Field(default="localhost", description="Host to bind the service to")
    port: int = Field(default=8000, description="Port to bind the service to", gt=0, lt=65536)
    log_level: str = Field(default="INFO", description="Logging level")
    environment: str = Field(default="development", description="Environment (development, test, production)")
    max_file_size_mb: int = Field(default=25, description="Maximum file size in MB", gt=0)
    allowed_file_types: List[str] = Field(
        default=["audio/wav", "audio/mp3", "audio/mpeg"],
        description="List of allowed file types"
    )
    temp_dir: str = Field(
        default="/tmp/whisper/temp",
        description="Temporary directory for file uploads"
    )
    cleanup_interval_minutes: int = Field(default=60, description="Interval for cleaning up temporary files", gt=0)
    json_logs: bool = Field(default=True, description="Whether to output logs in JSON format")
    batch_size: int = Field(default=8, description="Batch size for processing", gt=0)
    request_timeout: int = Field(default=300, description="Request timeout in seconds", gt=0)

    @field_validator("environment", mode="before")
    def validate_environment(cls, v: str) -> str:
        """Validate environment."""
        valid_envs = ["development", "test", "production"]
        v = str(v).lower()
        if v not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v

    @field_validator("log_level", mode="before")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = str(v).upper()
        if v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v

    @field_validator("temp_dir", mode="before")
    def validate_temp_dir(cls, v: str) -> str:
        """Validate and create temp directory."""
        v = os.path.expanduser(str(v))
        if not os.path.isabs(v):
            v = os.path.abspath(v)
        try:
            os.makedirs(v, exist_ok=True)
        except OSError:
            # If we can't create the directory, just return the path
            # The model validator will check if it's writable later
            pass
        return v

    @model_validator(mode='after')
    def validate_service(self) -> 'ServiceConfig':
        """Validate service configuration."""
        # Ensure temp directory exists and is writable
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
            if not os.access(self.temp_dir, os.W_OK):
                raise ValueError(f"Temporary directory {self.temp_dir} is not writable")
        except OSError:
            # If we can't create or access the directory, use a fallback
            self.temp_dir = "/tmp/whisper/temp"
            os.makedirs(self.temp_dir, exist_ok=True)
        return self

    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return ModelConfig(
            whisper_model=WhisperModel.BASE,
            device=DeviceType.CPU,
            compute_type=ComputeType.FLOAT32,
            download_root=os.path.join(os.path.dirname(self.temp_dir), "models"),
            language="en",
            cpu_threads=4,
            num_workers=2
        )

    def get_allowed_formats(self) -> List[str]:
        """Get list of allowed MIME types."""
        return self.allowed_file_types

    def get_temp_dir(self) -> str:
        """Get temporary directory path."""
        os.makedirs(self.temp_dir, exist_ok=True)
        return self.temp_dir

# Create global config instance
config = ServiceConfig() 
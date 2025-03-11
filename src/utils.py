from typing import List, Optional, Dict, Tuple
import os
from fastapi import HTTPException, UploadFile
import magic
import logging
import shutil
from datetime import datetime, timedelta
import asyncio
import aiofiles
import hashlib
from enum import Enum
from src.logging_setup import get_logger, error_tracker
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

# Constants
TEMP_DIR = "/tmp/whisper_audio"
CHUNK_SIZE = 8192  # 8KB chunks for file operations

class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    M4A = "m4a"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ALLOWED_AUDIO_TYPES = [
    'audio/wav',
    'audio/x-wav',
    'audio/mp3',
    'audio/mpeg',
    'audio/ogg',
    'audio/x-m4a',
    'audio/aac'
]

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB

class AudioValidationError(Exception):
    """Custom exception for audio validation errors."""
    pass

class AudioMetadata(BaseModel):
    """Audio file metadata."""
    original_filename: str
    temp_path: str
    file_type: str
    file_size: int
    content_hash: str
    duration: Optional[float] = None

    def __str__(self) -> str:
        return f"AudioMetadata({self.original_filename}, {self.file_type}, {self.file_size} bytes)"

    @field_validator('temp_path')
    @classmethod
    def validate_temp_path(cls, v):
        """Convert Path objects to strings and validate path."""
        if isinstance(v, (str, Path)):
            return str(v)
        raise ValueError("temp_path must be a string or Path object")

class AudioValidator:
    """Audio file validator with comprehensive checks."""
    
    def __init__(self, config: Dict):
        self.max_file_size = config.get('max_file_size', 25 * 1024 * 1024)  # 25MB default
        self.allowed_types = config.get('allowed_audio_types', [
            'audio/wav', 'audio/x-wav',
            'audio/mp3', 'audio/mpeg',
            'audio/ogg', 'audio/x-m4a',
            'audio/aac'
        ])
        self.temp_dir = config.get('temp_dir', '/tmp/whisper_audio')
        self.logger = get_logger(__name__)

    async def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file contents."""
        sha256_hash = hashlib.sha256()
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    async def save_upload_file(self, upload_file: UploadFile) -> str:
        """Save uploaded file with proper error handling."""
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = os.path.join(
                self.temp_dir,
                f"{timestamp}_{upload_file.filename}"
            )
            
            async with aiofiles.open(temp_file, "wb") as buffer:
                while content := await upload_file.read(8192):
                    await buffer.write(content)
                    
            self.logger.info("file_saved", 
                           temp_path=temp_file,
                           original_filename=upload_file.filename)
            return temp_file
            
        except Exception as e:
            error_tracker.track_error(e, {
                'filename': upload_file.filename,
                'operation': 'save_upload_file'
            })
            self.logger.error("file_save_failed",
                            error=str(e),
                            original_filename=upload_file.filename)
            raise HTTPException(
                status_code=500,
                detail="Failed to save uploaded file"
            )

    async def validate_audio_file(self, file_path: str | Path, original_filename: str) -> AudioMetadata:
        """Validate audio file with comprehensive checks."""
        try:
            # Convert Path to string if needed
            file_path_str = str(file_path)
            
            if not os.path.exists(file_path_str):
                raise AudioValidationError("File not found")
                
            # Check file size
            file_size = os.path.getsize(file_path_str)
            if file_size > self.max_file_size:
                raise AudioValidationError(
                    f"File size {file_size/1024/1024:.1f}MB exceeds maximum allowed size of {self.max_file_size/1024/1024}MB"
                )
            
            # Check file type
            file_type = magic.from_file(file_path_str, mime=True)
            if file_type not in self.allowed_types:
                raise AudioValidationError(
                    f"Invalid audio format: {file_type}. Allowed formats: {', '.join(self.allowed_types)}"
                )
            
            # Compute file hash
            content_hash = await self.compute_file_hash(file_path_str)
            
            metadata = AudioMetadata(
                original_filename=original_filename,
                temp_path=file_path_str,
                file_type=file_type,
                file_size=file_size,
                content_hash=content_hash
            )
            
            self.logger.info("file_validation_successful",
                           metadata=metadata.model_dump())
            
            return metadata
            
        except AudioValidationError as e:
            error_tracker.track_error(e, {
                'file_path': str(file_path),
                'original_filename': original_filename
            })
            self.logger.warning("file_validation_failed",
                              error=str(e),
                              file_path=str(file_path))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            error_tracker.track_error(e, {
                'file_path': str(file_path),
                'original_filename': original_filename
            })
            self.logger.error("unexpected_validation_error",
                            error=str(e),
                            file_path=str(file_path))
            raise HTTPException(
                status_code=500,
                detail="Error validating audio file"
            )

class TempFileManager:
    """Manage temporary files for the Whisper service."""

    def __init__(self, temp_dir: str = "/tmp/test_whisper_audio"):
        """Initialize the temporary file manager.
        
        Args:
            temp_dir: Directory to store temporary files
        """
        self.temp_dir = temp_dir
        self.logger = get_logger(__name__)
        
        # Ensure temp directory exists
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
            self.logger.info("temp_dir_initialized", path=self.temp_dir)
        except Exception as e:
            self.logger.error("temp_dir_initialization_failed", error=str(e))

    async def remove_temp_file(self, file_path: str) -> None:
        """Remove a temporary file.
        
        Args:
            file_path: Path to the file to remove
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.info("temp_file_removed", file_path=file_path)
        except Exception as e:
            self.logger.error("temp_file_removal_failed", error=str(e), file_path=file_path)

    async def cleanup_temp_files(self, max_age_hours: float = 24.0) -> None:
        """Clean up old temporary files.
        
        Args:
            max_age_hours: Maximum age of files to keep in hours
        """
        try:
            if not os.path.exists(self.temp_dir):
                self.logger.warning("temp_dir_not_found", path=self.temp_dir)
                return

            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if mtime < cutoff_time:
                            await self.remove_temp_file(file_path)
                except Exception as e:
                    self.logger.error("file_cleanup_failed", error=str(e), file_path=file_path)

        except Exception as e:
            self.logger.error("cleanup_failed", error=str(e))

    async def start_cleanup_scheduler(self, interval_hours: float = 1.0) -> None:
        """Start the cleanup scheduler.
        
        Args:
            interval_hours: Interval between cleanup runs in hours
        """
        while True:
            try:
                await self.cleanup_temp_files()
                await asyncio.sleep(interval_hours * 3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("cleanup_scheduler_error", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying

def validate_audio_file(file_path: str) -> None:
    """
    Validate audio file type and size.
    
    Args:
        file_path (str): Path to the audio file
        
    Raises:
        HTTPException: If file is too large or invalid format
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="File not found")
            
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File size {file_size/1024/1024:.1f}MB exceeds maximum allowed size of {MAX_FILE_SIZE/1024/1024}MB"
            )
        
        # Check file type
        file_type = magic.from_file(file_path, mime=True)
        if file_type not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid audio format: {file_type}. Allowed formats: {', '.join(ALLOWED_AUDIO_TYPES)}"
            )
            
        logger.info(f"File validation successful: {file_path}")
        
    except HTTPException as e:
        logger.error(f"Validation error: {str(e.detail)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during validation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error validating audio file")

def cleanup_temp_file(file_path: str) -> None:
    """
    Safely cleanup temporary files.
    
    Args:
        file_path (str): Path to the temporary file
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Temporary file cleaned up: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary file {file_path}: {str(e)}")

async def get_audio_duration(file_path: str) -> float:
    """Get the duration of an audio file in seconds.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Duration in seconds
        
    Raises:
        FileNotFoundError: If the file does not exist
        RuntimeError: If there is an error reading the file
    """
    try:
        with sf.SoundFile(file_path) as f:
            return float(len(f)) / f.samplerate
    except Exception as e:
        logger = get_logger(__name__)
        logger.error("Failed to get audio duration")
        raise

def create_temp_directory(path: str) -> str:
    """Create temporary directory if it doesn't exist."""
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except Exception as e:
        logger.error("Failed to create temp directory", exc_info=True)
        raise

def remove_temp_directory(path: str) -> None:
    """Remove temporary directory and its contents."""
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except Exception as e:
        logger.error("Failed to remove temp directory", error=str(e))

def get_mime_type(file_path: str) -> str:
    """Get MIME type of a file."""
    try:
        return magic.from_file(file_path, mime=True)
    except Exception as e:
        logger.error("Failed to get MIME type", exc_info=True)
        raise
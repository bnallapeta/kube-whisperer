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
import soundfile as sf
from src.logging_setup import get_logger, error_tracker
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import time
import structlog
from src.config import TranscriptionOptions
import psutil

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
    'audio/wav', 'audio/x-wav', 'audio/mp3', 'audio/mpeg',
    'audio/ogg', 'audio/x-m4a', 'audio/aac', 'audio/flac'
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
        self.allowed_types = config.get('allowed_types', ALLOWED_AUDIO_TYPES)
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
        os.makedirs(temp_dir, exist_ok=True)
        self._cleanup_task = None
        self.logger = structlog.get_logger(__name__)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_cleanup_scheduler()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_cleanup_scheduler()

    async def remove_temp_file(self, file_path: str) -> None:
        """Remove a temporary file."""
        try:
            if os.path.exists(file_path):
                await asyncio.to_thread(os.unlink, file_path)
                self.logger.info("temp_file_removed", file_path=file_path)
        except Exception as e:
            self.logger.error("temp_file_remove_failed", file_path=file_path, error=str(e))
            raise

    async def cleanup_temp_files(self, max_age_hours: int = 24) -> None:
        """Clean up temporary files older than max_age_hours."""
        try:
            current_time = time.time()
            max_age = max_age_hours * 3600  # Convert hours to seconds
            
            try:
                # Get list of files in temp directory
                files = await asyncio.to_thread(os.listdir, self.temp_dir)
            except FileNotFoundError:
                self.logger.warning("temp_dir_not_found", temp_dir=self.temp_dir)
                return
            
            # Process files in batches to avoid blocking
            batch_size = 10
            for i in range(0, len(files), batch_size):
                batch = files[i:i + batch_size]
                tasks = []
                
                for filename in batch:
                    file_path = os.path.join(self.temp_dir, filename)
                    try:
                        # Get file stats
                        stats = await asyncio.to_thread(os.stat, file_path)
                        file_age = current_time - stats.st_mtime
                        
                        if file_age > max_age:
                            tasks.append(self.remove_temp_file(file_path))
                    except Exception as e:
                        self.logger.error("file_age_check_failed", file_path=file_path, error=str(e))
                
                # Wait for batch to complete
                if tasks:
                    await asyncio.gather(*tasks)
                    
        except Exception as e:
            self.logger.error("cleanup_failed", error=str(e))
            raise

    async def start_cleanup_scheduler(self, interval_hours: int = 1) -> None:
        """Start the cleanup scheduler."""
        if self._cleanup_task is not None:
            return
            
        async def run_cleanup():
            while True:
                try:
                    await self.cleanup_temp_files()
                except Exception as e:
                    self.logger.error("scheduler_error", error=str(e))
                await asyncio.sleep(interval_hours * 3600)
        
        self._cleanup_task = asyncio.create_task(run_cleanup())

    async def stop_cleanup_scheduler(self) -> None:
        """Stop the cleanup scheduler."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

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

async def get_audio_duration(file_path: str | Path) -> float:
    """Get the duration of an audio file in seconds.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Duration in seconds
        
    Raises:
        FileNotFoundError: If the file does not exist
        LibsndfileError: If there is an error reading the file
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        with sf.SoundFile(file_path) as f:
            return float(len(f)) / f.samplerate
            
    except Exception as e:
        error_tracker.track_error(e, {'file_path': str(file_path)})
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

def process_audio(audio_path: str, options: Optional[TranscriptionOptions] = None) -> dict:
    """Process audio file with Whisper model."""
    try:
        import whisper
        model = whisper.load_model("base")
        opts = options.model_dump() if options else {}
        result = model.transcribe(audio_path, **opts)
        return result
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {str(e)}")
        raise e
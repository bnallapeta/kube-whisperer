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
from logging_setup import get_logger, error_tracker
from pydantic import BaseModel, Field

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
    """Metadata for validated audio files."""
    original_filename: str
    temp_path: str
    file_type: str
    file_size: int
    content_hash: str
    duration: Optional[float] = None

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

    async def validate_audio_file(self, file_path: str, original_filename: str) -> AudioMetadata:
        """Validate audio file with comprehensive checks."""
        try:
            if not os.path.exists(file_path):
                raise AudioValidationError("File not found")
                
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                raise AudioValidationError(
                    f"File size {file_size/1024/1024:.1f}MB exceeds maximum allowed size of {self.max_file_size/1024/1024}MB"
                )
            
            # Check file type
            file_type = magic.from_file(file_path, mime=True)
            if file_type not in self.allowed_types:
                raise AudioValidationError(
                    f"Invalid audio format: {file_type}. Allowed formats: {', '.join(self.allowed_types)}"
                )
            
            # Compute file hash
            content_hash = await self.compute_file_hash(file_path)
            
            metadata = AudioMetadata(
                original_filename=original_filename,
                temp_path=file_path,
                file_type=file_type,
                file_size=file_size,
                content_hash=content_hash
            )
            
            self.logger.info("file_validation_successful",
                           metadata=metadata.model_dump())
            
            return metadata
            
        except AudioValidationError as e:
            error_tracker.track_error(e, {
                'file_path': file_path,
                'original_filename': original_filename
            })
            self.logger.warning("file_validation_failed",
                              error=str(e),
                              file_path=file_path)
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            error_tracker.track_error(e, {
                'file_path': file_path,
                'original_filename': original_filename
            })
            self.logger.error("unexpected_validation_error",
                            error=str(e),
                            file_path=file_path)
            raise HTTPException(
                status_code=500,
                detail="Error validating audio file"
            )

class TempFileManager:
    """Manage temporary files with proper cleanup."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.logger = get_logger(__name__)
        self._cleanup_task = None

    async def cleanup_temp_files(self, max_age_hours: int = 24) -> None:
        """Cleanup temporary files older than max_age_hours."""
        try:
            if not os.path.exists(self.temp_dir):
                return
                
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                try:
                    if not os.path.isfile(file_path):
                        continue
                        
                    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if mtime < cutoff_time:
                        os.remove(file_path)
                        self.logger.info("temp_file_removed",
                                      file_path=file_path,
                                      age_hours=(datetime.now() - mtime).total_seconds() / 3600)
                                  
                except Exception as e:
                    error_tracker.track_error(e, {
                        'operation': 'cleanup_temp_files',
                        'file_path': file_path
                    })
                    self.logger.error("failed_to_remove_temp_file",
                                   error=str(e),
                                   file_path=file_path)
                    
        except Exception as e:
            error_tracker.track_error(e, {
                'operation': 'cleanup_temp_files',
                'temp_dir': self.temp_dir
            })
            self.logger.error("temp_cleanup_failed",
                            error=str(e),
                            temp_dir=self.temp_dir)

    async def start_cleanup_scheduler(self, interval_hours: int = 1) -> None:
        """Start periodic cleanup of temporary files."""
        while True:
            await self.cleanup_temp_files()
            await asyncio.sleep(interval_hours * 3600)

    def remove_temp_file(self, file_path: str) -> None:
        """Safely remove a temporary file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.info("temp_file_removed", file_path=file_path)
        except Exception as e:
            error_tracker.track_error(e, {
                'operation': 'remove_temp_file',
                'file_path': file_path
            })
            self.logger.error("failed_to_remove_temp_file",
                           error=str(e),
                           file_path=file_path)

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
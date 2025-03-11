from typing import List
import os
from fastapi import HTTPException
import magic
import logging

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
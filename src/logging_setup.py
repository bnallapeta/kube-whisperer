import structlog
import logging
import sys
import time
from typing import Any, Dict, Optional
import uuid
from contextvars import ContextVar
import json
from datetime import datetime
from collections import defaultdict
import traceback
import os

# Context variables for request tracking
request_id_context: ContextVar[str] = ContextVar('request_id', default='')
correlation_id_context: ContextVar[str] = ContextVar('correlation_id', default='')
session_id_context: ContextVar[str] = ContextVar('session_id', default='')

def set_context(request_id: Optional[str] = None, correlation_id: Optional[str] = None, 
              session_id: Optional[str] = None, environment: Optional[str] = None, 
              service: Optional[str] = None) -> Dict[str, str]:
    """Set context variables for request tracking."""
    context = {}
    
    if request_id:
        request_id_context.set(request_id)
        context['request_id'] = request_id
    else:
        new_request_id = str(uuid.uuid4())
        request_id_context.set(new_request_id)
        context['request_id'] = new_request_id

    if correlation_id:
        correlation_id_context.set(correlation_id)
        context['correlation_id'] = correlation_id
    
    if session_id:
        session_id_context.set(session_id)
        context['session_id'] = session_id
        
    if environment:
        context['environment'] = environment
        
    if service:
        context['service'] = service
    
    return context

def get_context() -> Dict[str, str]:
    """Get all context variables."""
    return {
        'request_id': request_id_context.get(),
        'correlation_id': correlation_id_context.get(),
        'session_id': session_id_context.get()
    }

def add_context_to_event(logger: Any, method_name: str, event_dict: Dict) -> Dict:
    """Add context variables to log event."""
    context = get_context()
    event_dict.update({k: v for k, v in context.items() if v})
    return event_dict

def add_timestamp(logger: Any, method_name: str, event_dict: Dict) -> Dict:
    """Add ISO format timestamp to log event."""
    event_dict['timestamp'] = datetime.utcnow().isoformat() + 'Z'
    return event_dict

def add_service_info(logger: Any, method_name: str, event_dict: Dict) -> Dict:
    """Add service information to log event."""
    event_dict['service'] = 'whisper-service'
    event_dict['environment'] = os.getenv('ENVIRONMENT', 'development')
    return event_dict

def format_exc_info(logger: Any, method_name: str, event_dict: Dict) -> Dict:
    """Format exception information in a structured way."""
    if 'exc_info' in event_dict and event_dict['exc_info']:
        if isinstance(event_dict['exc_info'], tuple):
            exc_type, exc_value, exc_tb = event_dict['exc_info']
            event_dict['error'] = {
                'type': exc_type.__name__,
                'message': str(exc_value),
                'traceback': traceback.format_tb(exc_tb)
            }
        del event_dict['exc_info']
    return event_dict

def setup_logging(log_level: str = "INFO", json_output: bool = True) -> None:
    """Configure logging with structlog.
    
    Args:
        log_level: The log level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Whether to output logs in JSON format
    """
    # Set environment variables
    os.environ.setdefault("ENVIRONMENT", "development")
    os.environ.setdefault("REQUEST_ID", "")
    os.environ.setdefault("CORRELATION_ID", "")
    os.environ.setdefault("SESSION_ID", "")

    # Convert string log level to numeric value
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level)

    # Configure processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance with the given name.
    
    Args:
        name: The name of the logger
        
    Returns:
        A configured logger instance
    """
    logger = structlog.get_logger(name)
    logger = logger.bind(
        environment=os.getenv("ENVIRONMENT", "development"),
        request_id=os.getenv("REQUEST_ID", ""),
        correlation_id=os.getenv("CORRELATION_ID", ""),
        session_id=os.getenv("SESSION_ID", "")
    )
    return logger

# Error tracking
class ErrorTracker:
    """Track and log errors with a maximum count."""
    
    def __init__(self, max_errors: int = 100):
        self.max_errors = max_errors
        self.error_counts = {}
        self.logger = get_logger("error_tracker")

    def track_error(self, error: Exception, context: Optional[dict] = None) -> None:
        """Track an error occurrence."""
        error_type = type(error).__name__
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        if self.error_counts[error_type] <= self.max_errors:
            error_info = {
                "type": error_type,
                "message": str(error),
                "timestamp": datetime.utcnow().isoformat(),
                "context": context or {}
            }
            self.logger.error("error_tracked", 
                            error_type=error_type,
                            error_count=self.error_counts[error_type],
                            error_info=error_info)

error_tracker = ErrorTracker()
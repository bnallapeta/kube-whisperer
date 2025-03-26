import structlog
import logging
import sys
import time
from typing import Any, Dict, Optional
import uuid
from contextvars import ContextVar, Token
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
              service: Optional[str] = None) -> Dict[str, Any]:
    """Set context variables for request tracking."""
    context = {}
    tokens = {}
    
    if request_id:
        tokens['request_id'] = request_id_context.set(request_id)
        context['request_id'] = request_id
        os.environ["REQUEST_ID"] = request_id
    else:
        new_request_id = str(uuid.uuid4())
        tokens['request_id'] = request_id_context.set(new_request_id)
        context['request_id'] = new_request_id
        os.environ["REQUEST_ID"] = new_request_id

    if correlation_id:
        tokens['correlation_id'] = correlation_id_context.set(correlation_id)
        context['correlation_id'] = correlation_id
        os.environ["CORRELATION_ID"] = correlation_id
    else:
        new_correlation_id = "corr-" + str(uuid.uuid4())[:8]
        tokens['correlation_id'] = correlation_id_context.set(new_correlation_id)
        context['correlation_id'] = new_correlation_id
        os.environ["CORRELATION_ID"] = new_correlation_id
    
    if session_id:
        tokens['session_id'] = session_id_context.set(session_id)
        context['session_id'] = session_id
        os.environ["SESSION_ID"] = session_id
    else:
        new_session_id = "sess-" + str(uuid.uuid4())[:8]
        tokens['session_id'] = session_id_context.set(new_session_id)
        context['session_id'] = new_session_id
        os.environ["SESSION_ID"] = new_session_id
        
    if environment:
        os.environ["ENVIRONMENT"] = environment
        context['environment'] = environment
    else:
        os.environ["ENVIRONMENT"] = "development"
        context['environment'] = "development"
    
    if service:
        os.environ["SERVICE_NAME"] = service
        context['service'] = service
    else:
        os.environ["SERVICE_NAME"] = "whisper-service"
        context['service'] = "whisper-service"
    
    context['tokens'] = tokens
    return context

def restore_context(context: Dict[str, Any]) -> None:
    """Restore context variables using tokens."""
    if 'tokens' in context:
        tokens = context['tokens']
        for var_name, token in tokens.items():
            if var_name == 'request_id':
                request_id_context.reset(token)
                os.environ["REQUEST_ID"] = context.get('request_id', '')
            elif var_name == 'correlation_id':
                correlation_id_context.reset(token)
                os.environ["CORRELATION_ID"] = context.get('correlation_id', '')
            elif var_name == 'session_id':
                session_id_context.reset(token)
                os.environ["SESSION_ID"] = context.get('session_id', '')
        
        # Restore environment and service if they were set
        if 'environment' in context:
            os.environ["ENVIRONMENT"] = context['environment']
        if 'service' in context:
            os.environ["SERVICE_NAME"] = context['service']

def get_context() -> Dict[str, str]:
    """Get all context variables."""
    return {
        'request_id': request_id_context.get(),
        'correlation_id': correlation_id_context.get(),
        'session_id': session_id_context.get(),
        'environment': os.getenv('ENVIRONMENT', 'development'),
        'service': os.getenv('SERVICE_NAME', 'whisper-service')
    }

def add_context(logger, method_name, event_dict):
    """Add context variables to the event dictionary."""
    event_dict['request_id'] = request_id_context.get()
    event_dict['correlation_id'] = correlation_id_context.get()
    event_dict['session_id'] = session_id_context.get()
    event_dict['environment'] = os.getenv("ENVIRONMENT", "development")
    event_dict['service'] = os.getenv("SERVICE_NAME", "whisper-service")
    return event_dict

def add_timestamp(logger, method_name, event_dict):
    """Add timestamp to the event dictionary."""
    event_dict['timestamp'] = datetime.utcnow().isoformat()
    return event_dict

def add_service_info(logger: Any, method_name: str, event_dict: Dict) -> Dict:
    """Add service information to log event."""
    event_dict['service'] = os.getenv('SERVICE_NAME', 'whisper-service')
    event_dict['environment'] = os.getenv('ENVIRONMENT', 'development')
    return event_dict

def add_log_level(logger, method_name, event_dict):
    """Add log level to the event dictionary."""
    event_dict['log_level'] = method_name.upper()
    return event_dict

def add_request_id(logger, method_name, event_dict):
    """Add request ID to the event dictionary."""
    event_dict['request_id'] = request_id_context.get()
    return event_dict

def add_correlation_id(logger, method_name, event_dict):
    """Add correlation ID to the event dictionary."""
    event_dict['correlation_id'] = correlation_id_context.get()
    return event_dict

def add_session_id(logger, method_name, event_dict):
    """Add session ID to the event dictionary."""
    event_dict['session_id'] = session_id_context.get()
    return event_dict

def add_environment(logger, method_name, event_dict):
    """Add environment to the event dictionary."""
    event_dict['environment'] = os.getenv("ENVIRONMENT", "development")
    return event_dict

def add_service_name(logger, method_name, event_dict):
    """Add service name to the event dictionary."""
    event_dict['service'] = os.getenv("SERVICE_NAME", "whisper-service")
    return event_dict

def add_standard_log_level(logger, method_name, event_dict):
    """Add standard log level to the event dictionary."""
    level_map = {
        'debug': 'DEBUG',
        'info': 'INFO',
        'warning': 'WARNING',
        'error': 'ERROR',
        'critical': 'CRITICAL'
    }
    event_dict['level'] = level_map.get(method_name.lower(), 'INFO')
    return event_dict

def setup_logging(log_level: str = "INFO", json_output: bool = False) -> None:
    """Configure logging with structured output."""
    # Set default context if not already set
    if not request_id_context.get():
        set_context()
    
    # Convert string log level to numeric value
    numeric_level = getattr(logging, log_level.upper())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add new handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    root_logger.addHandler(handler)
    
    # Configure processors
    processors = [
        add_standard_log_level,
        add_log_level,
        add_context,
        add_timestamp,
        add_request_id,
        add_correlation_id,
        add_session_id,
        add_environment,
        add_service_name,
    ]
    
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
        context_class=dict,
    )

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance with the given name.
    
    Args:
        name: The name of the logger
        
    Returns:
        A configured logger instance
    """
    logger = structlog.get_logger(name)
    context = get_context()
    logger = logger.bind(**context)
    return logger

# Error tracking
class ErrorTracker:
    def __init__(self, max_errors: int = 100):
        self.max_errors = max_errors
        self.errors = []
        self.error_count = 0
        self.logger = get_logger("error_tracker")

    def track_error(self, error: Exception, context: Dict = None) -> None:
        """Track an error with context."""
        error_info = {
            'type': error.__class__.__name__,
            'message': str(error),
            'timestamp': datetime.utcnow().isoformat(),
            'context': context or {},
            'traceback': ''.join(traceback.format_tb(error.__traceback__))
        }
        
        self.error_count += 1
        self.errors.append(error_info)
        
        # Keep only the most recent errors
        if len(self.errors) > self.max_errors:
            self.errors.pop(0)
        
        # Log the error
        self.logger.error("error_tracked",
                         error_type=error_info['type'],
                         error_count=self.error_count,
                         error_info=error_info)

error_tracker = ErrorTracker()
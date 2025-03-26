from datetime import datetime
from src.logging_setup import (
    setup_logging,
    add_context,
    add_log_level,
    add_timestamp,
    add_request_id,
    add_correlation_id,
    add_session_id,
    add_environment,
    add_service_name,
    add_standard_log_level,
    set_context,
    restore_context,
    request_id_context,
    correlation_id_context,
    session_id_context,
    get_logger,
    ErrorTracker,
)

import pytest
import logging
import json
import os
from unittest.mock import patch, MagicMock

def test_context_variables():
    """Test context variables are set correctly."""
    request_id = "test-request-id"
    correlation_id = "test-correlation-id"
    session_id = "test-session-id"
    
    context = set_context(request_id, correlation_id, session_id)
    
    assert request_id_context.get() == request_id
    assert correlation_id_context.get() == correlation_id
    assert session_id_context.get() == session_id

def test_context_management():
    """Test context management."""
    request_id = "test-request-id"
    correlation_id = "test-correlation-id"
    session_id = "test-session-id"
    
    context = set_context(request_id, correlation_id, session_id)
    
    # Test context restoration
    new_context = set_context("new-request-id", "new-correlation-id", "new-session-id")
    restore_context(context)
    
    assert request_id_context.get() == request_id
    assert correlation_id_context.get() == correlation_id
    assert session_id_context.get() == session_id

def test_logger_setup_with_different_levels():
    """Test logger setup with different configurations."""
    # Test DEBUG level
    setup_logging(log_level="DEBUG", json_output=True)
    logger = get_logger(__name__)
    root_logger = logging.getLogger()

    mock_stream = MagicMock()
    root_logger.handlers[0].stream = mock_stream
    logger.debug("test message")
    assert len(mock_stream.write.call_args_list) > 0
    output = mock_stream.write.call_args_list[0][0][0]
    log_data = json.loads(output)
    assert log_data['event'] == "test message"
    assert log_data['level'] == "DEBUG"

def test_logger_output_formats():
    """Test different logger output formats."""
    # Test JSON output
    setup_logging(log_level="INFO", json_output=True)
    logger = get_logger(__name__)
    root_logger = logging.getLogger()

    mock_stream = MagicMock()
    root_logger.handlers[0].stream = mock_stream
    logger.info("test message")
    assert len(mock_stream.write.call_args_list) > 0
    output = mock_stream.write.call_args_list[0][0][0]
    log_data = json.loads(output)
    assert log_data['event'] == "test message"
    assert log_data['level'] == "INFO"

def test_error_tracker_with_different_errors():
    """Test error tracker with different error types."""
    error_tracker = ErrorTracker()

    try:
        raise ValueError("test error")
    except Exception as e:
        error_tracker.track_error(e, {'test': 'data'})

    assert len(error_tracker.errors) == 1
    error = error_tracker.errors[0]
    assert error['type'] == 'ValueError'
    assert error['message'] == 'test error'
    assert error['context'] == {'test': 'data'}

@pytest.mark.asyncio
async def test_logging_with_nested_context():
    """Test logging with nested context variables."""
    setup_logging(log_level="DEBUG", json_output=True)
    logger = get_logger(__name__)
    root_logger = logging.getLogger()

    # Set outer context
    outer_context = set_context("outer-request", "outer-correlation", "outer-session")

    mock_stream = MagicMock()
    root_logger.handlers[0].stream = mock_stream
    logger.info("outer message")
    assert len(mock_stream.write.call_args_list) > 0
    output = mock_stream.write.call_args_list[0][0][0]
    log_data = json.loads(output)
    assert log_data['event'] == "outer message"
    assert log_data['request_id'] == "outer-request"

def test_error_tracker_max_errors():
    """Test error tracker max errors limit."""
    error_tracker = ErrorTracker(max_errors=2)
    
    for i in range(3):
        try:
            raise ValueError(f"test error {i}")
        except Exception as e:
            error_tracker.track_error(e, {})
    
    assert len(error_tracker.errors) == 2

def test_log_processors_chain():
    """Test log processors chain."""
    setup_logging(log_level="INFO", json_output=True)
    logger = get_logger(__name__)
    root_logger = logging.getLogger()

    mock_stream = MagicMock()
    root_logger.handlers[0].stream = mock_stream
    logger.info("test message")
    assert len(mock_stream.write.call_args_list) > 0
    output = mock_stream.write.call_args_list[0][0][0]
    log_data = json.loads(output)
    assert log_data['event'] == "test message"
    assert log_data['level'] == "INFO"

def test_logger_with_environment_vars(monkeypatch):
    """Test logging with environment variables."""
    monkeypatch.setenv("ENVIRONMENT", "production")

    setup_logging(log_level="INFO", json_output=True)
    logger = get_logger(__name__)
    root_logger = logging.getLogger()

    mock_stream = MagicMock()
    root_logger.handlers[0].stream = mock_stream
    logger.info("test message")
    assert len(mock_stream.write.call_args_list) > 0
    output = mock_stream.write.call_args_list[0][0][0]
    log_data = json.loads(output)
    assert log_data['event'] == "test message"
    assert log_data['environment'] == "production"

def test_error_tracker_with_exception_info():
    """Test error tracker with exception info."""
    error_tracker = ErrorTracker()

    try:
        raise ValueError("test error")
    except Exception as e:
        error_tracker.track_error(e, {})

    error = error_tracker.errors[0]
    assert error['type'] == 'ValueError'
    assert error['message'] == 'test error'
    assert isinstance(error['traceback'], str)

def test_logger_with_stack_info():
    """Test logging with stack information."""
    setup_logging(log_level="DEBUG", json_output=True)
    logger = get_logger(__name__)
    root_logger = logging.getLogger()

    mock_stream = MagicMock()
    root_logger.handlers[0].stream = mock_stream
    logger.info("test message", stack_info=True)
    assert len(mock_stream.write.call_args_list) > 0
    output = mock_stream.write.call_args_list[0][0][0]
    log_data = json.loads(output)
    assert log_data['event'] == "test message"
    assert log_data['stack_info'] is True

def test_logger_with_different_log_levels():
    """Test logging at different levels."""
    setup_logging(log_level="DEBUG", json_output=True)
    logger = get_logger(__name__)
    root_logger = logging.getLogger()

    mock_stream = MagicMock()
    root_logger.handlers[0].stream = mock_stream
    logger.debug("debug message")
    assert len(mock_stream.write.call_args_list) > 0
    output = mock_stream.write.call_args_list[0][0][0]
    log_data = json.loads(output)
    assert log_data['event'] == "debug message"
    assert log_data['level'] == "DEBUG"

    logger.info("info message")
    output = mock_stream.write.call_args_list[1][0][0]
    log_data = json.loads(output)
    assert log_data['event'] == "info message"
    assert log_data['level'] == "INFO"

    logger.warning("warning message")
    output = mock_stream.write.call_args_list[2][0][0]
    log_data = json.loads(output)
    assert log_data['event'] == "warning message"
    assert log_data['level'] == "WARNING"

    logger.error("error message")
    output = mock_stream.write.call_args_list[3][0][0]
    log_data = json.loads(output)
    assert log_data['event'] == "error message"
    assert log_data['level'] == "ERROR"
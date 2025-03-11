import pytest
import structlog
import json
import logging
import sys
import os
from datetime import datetime
from src.logging_setup import (
    set_context, get_context, setup_logging,
    get_logger, ErrorTracker, add_context_to_event,
    add_timestamp, add_service_info, format_exc_info,
    request_id_context, correlation_id_context, session_id_context
)

def test_context_variables():
    """Test context variables initialization."""
    assert request_id_context.get() == ''
    assert correlation_id_context.get() == ''
    assert session_id_context.get() == ''

def test_context_management():
    """Test request context management."""
    # Test setting context
    context = set_context(
        request_id="test-123",
        correlation_id="corr-456",
        session_id="sess-789"
    )
    assert context["request_id"] == "test-123"
    assert context["correlation_id"] == "corr-456"
    assert context["session_id"] == "sess-789"

    # Test getting context
    current_context = get_context()
    assert current_context["request_id"] == "test-123"
    assert current_context["correlation_id"] == "corr-456"
    assert current_context["session_id"] == "sess-789"

    # Test auto-generated request ID
    new_context = set_context()
    assert "request_id" in new_context
    assert len(new_context["request_id"]) > 0

def test_logger_setup_with_different_levels():
    """Test logger setup with different configurations."""
    # Test DEBUG level
    setup_logging(log_level="DEBUG", json_output=True)
    assert logging.getLogger().level == logging.DEBUG

    # Test INFO level
    setup_logging(log_level="INFO", json_output=False)
    assert logging.getLogger().level == logging.INFO

    # Test WARNING level
    setup_logging(log_level="WARNING", json_output=True)
    assert logging.getLogger().level == logging.WARNING

def test_logger_output_formats():
    """Test different logger output formats."""
    # Test JSON output
    setup_logging(log_level="INFO", json_output=True)
    logger = get_logger("test_json")
    with structlog.testing.capture_logs() as captured:
        logger.info("test_json_message", test_field="test_value")
    assert len(captured) == 1
    assert isinstance(json.dumps(captured[0]), str)

    # Test console output
    setup_logging(log_level="INFO", json_output=False)
    logger = get_logger("test_console")
    with structlog.testing.capture_logs() as captured:
        logger.info("test_console_message", test_field="test_value")
    assert len(captured) == 1

def test_error_tracker_with_different_errors():
    """Test error tracker with various error types."""
    tracker = ErrorTracker()
    
    # Track different types of errors
    errors = [
        ValueError("Value error"),
        TypeError("Type error"),
        RuntimeError("Runtime error"),
        KeyError("Key error")
    ]
    
    for error in errors:
        tracker.track_error(error, {"error_type": error.__class__.__name__})
    
    stats = tracker.get_error_stats()
    
    # Check counts for each error type
    for error in errors:
        error_type = error.__class__.__name__
        assert error_type in stats["counts"]
        assert stats["counts"][error_type] == 1
        assert len(stats["recent_errors"][error_type]) == 1

def test_error_tracker_context_handling():
    """Test error tracker context handling."""
    tracker = ErrorTracker()
    test_error = ValueError("Test error")
    
    # Test with different contexts
    contexts = [
        {"user_id": "123", "action": "test1"},
        {"user_id": "456", "action": "test2"},
        {"request_id": "789", "endpoint": "/test"}
    ]
    
    for context in contexts:
        tracker.track_error(test_error, context)
    
    stats = tracker.get_error_stats()
    recent_errors = stats["recent_errors"]["ValueError"]
    
    # Verify contexts are preserved
    for i, context in enumerate(contexts):
        error_entry = recent_errors[i]
        assert error_entry["context"] == context

@pytest.mark.asyncio
async def test_logging_with_nested_context():
    """Test logging with nested context variables."""
    setup_logging(log_level="DEBUG", json_output=True)
    logger = get_logger("test_nested_context")
    
    # Set outer context
    outer_context = set_context(request_id="outer-req-id")
    
    with structlog.testing.capture_logs() as captured:
        logger.info("outer_message")
        
        # Set inner context
        inner_context = set_context(request_id="inner-req-id")
        logger.info("inner_message")
        
        # Reset to outer context
        set_context(request_id=outer_context["request_id"])
        logger.info("final_message")
    
    assert len(captured) == 3
    assert captured[0]["request_id"] == "outer-req-id"
    assert captured[1]["request_id"] == "inner-req-id"
    assert captured[2]["request_id"] == "outer-req-id"

def test_error_tracker_max_errors():
    """Test error tracker maximum error storage."""
    tracker = ErrorTracker()
    max_errors = tracker.max_stored_errors
    
    # Generate more errors than the maximum
    for i in range(max_errors + 50):
        tracker.track_error(
            ValueError(f"Error {i}"),
            {"iteration": i}
        )
    
    stats = tracker.get_error_stats()
    assert len(stats["recent_errors"]["ValueError"]) == max_errors
    # Verify we keep the most recent errors
    last_error = stats["recent_errors"]["ValueError"][-1]
    assert last_error["context"]["iteration"] == max_errors + 49

def test_log_processors_chain():
    """Test the complete chain of log processors."""
    event_dict = {}
    
    # Apply processors in sequence
    event_dict = add_context_to_event(None, "test", event_dict)
    event_dict = add_timestamp(None, "test", event_dict)
    event_dict = add_service_info(None, "test", event_dict)
    
    # Add an exception
    try:
        raise ValueError("Test exception")
    except Exception as e:
        event_dict["exc_info"] = (type(e), e, e.__traceback__)
    
    event_dict = format_exc_info(None, "test", event_dict)
    
    # Verify all processors have added their data
    assert "request_id" in event_dict
    assert "timestamp" in event_dict
    assert "service" in event_dict
    assert "environment" in event_dict
    assert "error" in event_dict
    assert "exc_info" not in event_dict  # Should be removed after formatting

def test_logger_with_environment_vars(monkeypatch):
    """Test logging with environment variables."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    
    setup_logging(log_level="INFO", json_output=True)
    logger = get_logger("test_env")
    
    with structlog.testing.capture_logs() as captured:
        logger.info("test_message")
    
    assert len(captured) == 1
    assert captured[0]["environment"] == "production"
    assert captured[0]["service"] == "whisper-service"

def test_error_tracker_with_exception_info():
    """Test error tracker with full exception info."""
    tracker = ErrorTracker()
    
    try:
        # Create a nested exception
        try:
            raise ValueError("Inner error")
        except ValueError as e:
            raise RuntimeError("Outer error") from e
    except RuntimeError as e:
        tracker.track_error(e)
    
    stats = tracker.get_error_stats()
    error_entry = stats["recent_errors"]["RuntimeError"][0]
    
    assert error_entry["type"] == "RuntimeError"
    assert "Outer error" in error_entry["message"]
    assert isinstance(error_entry["timestamp"], str)

def test_logger_with_stack_info():
    """Test logging with stack information."""
    setup_logging(log_level="DEBUG", json_output=True)
    logger = get_logger("test_stack")
    
    with structlog.testing.capture_logs() as captured:
        logger.info("test_message", stack_info=True)
    
    assert len(captured) == 1
    assert "stack_info" in captured[0]

def test_logger_with_different_log_levels():
    """Test logging at different levels."""
    setup_logging(log_level="DEBUG", json_output=True)
    logger = get_logger("test_levels")
    
    with structlog.testing.capture_logs() as captured:
        logger.debug("debug_message")
        logger.info("info_message")
        logger.warning("warning_message")
        logger.error("error_message")
        logger.critical("critical_message")
    
    assert len(captured) == 5
    log_levels = [entry["level"] for entry in captured]
    assert "debug" in log_levels
    assert "info" in log_levels
    assert "warning" in log_levels
    assert "error" in log_levels
    assert "critical" in log_levels

def test_error_tracker_stats_format():
    """Test error tracker statistics format."""
    tracker = ErrorTracker()
    
    # Track some errors
    tracker.track_error(ValueError("Error 1"))
    tracker.track_error(ValueError("Error 2"))
    tracker.track_error(TypeError("Error 3"))
    
    stats = tracker.get_error_stats()
    
    # Check stats structure
    assert "counts" in stats
    assert "recent_errors" in stats
    assert isinstance(stats["counts"], dict)
    assert isinstance(stats["recent_errors"], dict)
    
    # Check error counts
    assert stats["counts"]["ValueError"] == 2
    assert stats["counts"]["TypeError"] == 1
    
    # Check recent errors format
    assert len(stats["recent_errors"]["ValueError"]) == 2
    assert len(stats["recent_errors"]["TypeError"]) == 1
    
    # Check error entry format
    error_entry = stats["recent_errors"]["ValueError"][0]
    assert "type" in error_entry
    assert "message" in error_entry
    assert "timestamp" in error_entry
    assert "context" in error_entry 
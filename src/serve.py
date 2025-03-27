import sys
import traceback
import psutil
import structlog
import shutil
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from contextlib import asynccontextmanager
import uvicorn
import whisper
import torch
import os
import logging
from src.utils import validate_audio_file, cleanup_temp_file
import uuid
from typing import Optional, List, Dict, Any
import time
from pydantic import BaseModel, Field, ConfigDict, field_validator
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
from prometheus_fastapi_instrumentator import Instrumentator
from datetime import datetime, timedelta
from src.config import ModelConfig, ServiceConfig, TranscriptionOptions, DeviceType, ComputeType
from src.logging_setup import get_logger, error_tracker, setup_logging

# Initialize FastAPI app
app = FastAPI()

# Configure logging
setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"), json_output=True)
logger = get_logger(__name__)

# Ensure temp directory exists
TEMP_DIR = "/tmp/whisper_audio"
os.makedirs(TEMP_DIR, exist_ok=True)
logger.info("temp_dir_initialized", path=TEMP_DIR)

# Initialize Prometheus metrics and instrumentation
REQUESTS = Counter("whisper_requests_total", "Total requests processed")
ERRORS = Counter("whisper_errors_total", "Total errors encountered", ["type"])
PROCESSING_TIME = Histogram("whisper_processing_seconds", "Time spent processing requests")
GPU_MEMORY = Gauge("whisper_gpu_memory_bytes", "GPU memory usage in bytes")
CPU_USAGE = Gauge("whisper_cpu_usage_percent", "CPU usage percentage")
MEMORY_USAGE = Gauge("whisper_memory_usage_bytes", "Memory usage in bytes")

# Model configuration
# Remove duplicate ModelConfig class and use the one from config.py

# Transcription options
class TranscriptionOptions(BaseModel):
    """Options for transcription."""
    language: Optional[str] = Field(default="en", description="Language code (e.g., 'en', 'es')")
    task: str = Field(default="transcribe", description="Task type (transcribe or translate)")
    beam_size: int = Field(default=5, ge=1, le=10, description="Beam size for beam search")
    patience: float = Field(default=1.0, ge=0.0, le=2.0, description="Beam search patience factor")
    temperature: List[float] = Field(
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        min_length=1,
        max_length=10,
        description="Temperature for sampling"
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        """Validate language code."""
        if v is None:
            return "en"  # Default to English
        if not isinstance(v, str):
            raise ValueError("Language code must be a string")
        if len(v) != 2:
            raise ValueError("Language code must be a 2-letter ISO code")
        return v.lower()

    @field_validator("task")
    @classmethod
    def validate_task(cls, v):
        """Validate task type."""
        valid_tasks = ["transcribe", "translate"]
        if v not in valid_tasks:
            raise ValueError(f"Task must be one of: {', '.join(valid_tasks)}")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature values."""
        if not all(0.0 <= t <= 1.0 for t in v):
            raise ValueError("Temperature values must be between 0.0 and 1.0")
        return v

# Batch request
class BatchRequest(BaseModel):
    """Batch transcription request."""
    files: List[str] = Field(..., min_length=1, max_length=10, description="List of file paths to transcribe")
    options: TranscriptionOptions = Field(default_factory=TranscriptionOptions)

    @field_validator("files")
    @classmethod
    def validate_files(cls, v):
        """Validate file paths."""
        if not v:
            raise ValueError("At least one file path must be provided")
        if len(v) > 10:
            raise ValueError("Maximum of 10 files allowed per batch")
        # Don't check file existence here - it will be checked during processing
        return v

# Global variables
# Initialize ModelConfig with environment variables
config = ModelConfig(
    whisper_model=os.getenv("WHISPER_MODEL", "base"),
    device=os.getenv("DEVICE", "cpu"),
    compute_type=os.getenv("COMPUTE_TYPE", "int8"),
    cpu_threads=int(os.getenv("CPU_THREADS", "4")),
    num_workers=int(os.getenv("NUM_WORKERS", "1")),
    download_root=os.getenv("MODEL_DOWNLOAD_ROOT", "/tmp/whisper_models")
)
model = None
executor = None

def update_metrics():
    """Update system metrics."""
    try:
        if torch.cuda.is_available():
            GPU_MEMORY.set(torch.cuda.memory_allocated())
        process = psutil.Process()
        CPU_USAGE.set(process.cpu_percent())
        MEMORY_USAGE.set(process.memory_info().rss)
    except Exception as e:
        logger.error("metrics_update_failed", error=str(e))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application."""
    global model, config, executor, health_checker
    try:
        logger.info("loading_model", config=config.__dict__)
        model = load_model(config)
        logger.info("model_loaded_successfully")
        
        # Initialize health checker with loaded model
        health_checker = HealthStatus()
        logger.info("health_checker_initialized")
        
        if config.device in ["cpu", "auto"] and not torch.cuda.is_available():
            torch.set_num_threads(config.cpu_threads)
            logger.info("cpu_threads_set", threads=config.cpu_threads)
        
        executor = ThreadPoolExecutor(max_workers=config.num_workers)
        logger.info("executor_initialized", workers=config.num_workers)
        
    except Exception as e:
        logger.error("startup_failed", 
                    error=str(e),
                    traceback=traceback.format_exc(),
                    error_type=type(e).__name__)
        raise
    yield
    if executor:
        executor.shutdown(wait=True)
        logger.info("executor_shutdown")

# Set up Prometheus instrumentation before any middleware
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.options("/{path:path}")
async def options_route(path: str):
    """Handle CORS preflight requests."""
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Credentials": "true",
    }
    return Response(status_code=200, headers=headers)

def load_model(model_config: ModelConfig) -> whisper.Whisper:
    """Load Whisper model with specified configuration."""
    try:
        if model_config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = model_config.device

        if device == "cuda":
            torch.set_float32_matmul_precision("high")
            if model_config.compute_type == "float16":
                torch.set_default_dtype(torch.float16)

        model = whisper.load_model(
            model_config.whisper_model,
            device=device,
        )
        return model
    except Exception as e:
        logger.error("model_load_failed",
                    error=str(e),
                    traceback=traceback.format_exc(),
                    config=model_config.__dict__)
        raise

class HealthStatus:
    """Health status checker for the Whisper service."""

    def __init__(self, model=None, gpu_available: bool = False, temp_dir: str = "/tmp/whisper_audio"):
        """Initialize health status checker."""
        self.model = model
        self.model_loaded = model is not None
        self.gpu_available = gpu_available
        self.temp_dir = temp_dir
        self.temp_dir_writable = False
        self.system_resources_ok = True
        self.last_check_time = None
        logger.info("Health status initialized", 
                   model_loaded=self.model_loaded,
                   gpu_available=self.gpu_available,
                   temp_dir=self.temp_dir)

    async def check_model_health(self) -> Dict[str, Any]:
        """Check model health."""
        try:
            if not self.model_loaded or self.model is None:
                logger.warning("Model not loaded")
                return {"status": "error", "message": "Model not loaded"}
            return {
                "status": "healthy",
                "message": "Model loaded and ready",
                "device": "cpu",
                "model_name": "base",
                "compute_type": "float32"
            }
        except Exception as e:
            logger.error("Model health check failed", error=str(e))
            return {"status": "error", "message": str(e)}

    async def check_gpu_health(self) -> Dict[str, Any]:
        """Check GPU health."""
        try:
            if not self.gpu_available:
                return {"status": "not_available", "message": "GPU not required"}
            
            import torch
            if not torch.cuda.is_available():
                return {"status": "not_available", "message": "GPU not available"}
            
            return {
                "status": "healthy",
                "message": "GPU available",
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count()
            }
        except Exception as e:
            logger.error(f"GPU health check failed: {e}")
            return {"status": "error", "message": str(e)}

    async def check_system_health(self) -> Dict[str, Any]:
        """Check system resources."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = "healthy"
            warnings = []
            
            if cpu_percent > 90:
                status = "warning"
                warnings.append("High CPU usage")
                
            if memory.percent > 90:
                status = "warning"
                warnings.append("High memory usage")
                
            if disk.percent > 90:
                status = "warning"
                warnings.append("Low disk space")
            
            return {
                "status": status,
                "message": "; ".join(warnings) if warnings else "System resources OK",
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "thread_count": psutil.cpu_count()
            }
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return {"status": "error", "message": str(e)}

    async def check_temp_directory(self) -> Dict[str, Any]:
        """Check temporary directory."""
        try:
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            
            # Try to create a test file
            test_file = os.path.join(self.temp_dir, f"test_{int(time.time())}.tmp")
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                self.temp_dir_writable = True
            except Exception as e:
                self.temp_dir_writable = False
                return {"status": "error", "message": f"Temp directory not writable: {e}"}
            
            # Check disk space
            disk_usage = psutil.disk_usage(self.temp_dir)
            if disk_usage.free < 1_000_000_000:  # 1GB
                return {
                    "status": "warning",
                    "message": "Low disk space in temp directory",
                    "free_space": disk_usage.free,
                    "total_space": disk_usage.total
                }
            
            return {
                "status": "healthy",
                "message": "Temp directory accessible",
                "path": self.temp_dir,
                "free_space": disk_usage.free,
                "total_space": disk_usage.total
            }
        except Exception as e:
            logger.error(f"Temp directory check failed: {e}")
            return {"status": "error", "message": str(e)}

    async def get_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        self.last_check_time = datetime.utcnow().isoformat()
        
        model_health = await self.check_model_health()
        gpu_health = await self.check_gpu_health()
        system_health = await self.check_system_health()
        temp_dir_health = await self.check_temp_directory()
        
        overall_status = "ok"
        if any(check["status"] == "error" for check in [model_health, gpu_health, system_health, temp_dir_health]):
            overall_status = "error"
        elif any(check["status"] == "warning" for check in [model_health, gpu_health, system_health, temp_dir_health]):
            overall_status = "warning"
        
        return {
            "status": overall_status,
            "timestamp": self.last_check_time,
            "checks": {
                "model": model_health,
                "gpu": gpu_health,
                "system": system_health,
                "temp_directory": temp_dir_health
            }
        }

# Initialize health checker
health_checker = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global health_checker, model, config
    try:
        # Initialize model config
        config = ModelConfig()
        model = load_model(config)
        
        # Initialize health checker with model and GPU status
        health_checker = HealthStatus(
            model=model,
            gpu_available=torch.cuda.is_available(),
            temp_dir=TEMP_DIR
        )
        logger.info("Health checker initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        raise e

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Whisper Transcription Service"}

@app.get("/health")
async def health_check():
    """Get comprehensive health status of the service."""
    try:
        if health_checker is None:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "error",
                    "message": "Health checker not initialized",
                    "checks": {
                        "model": {"status": "error", "message": "Model not loaded"},
                        "gpu": {"status": "not_available", "message": "GPU not required"},
                        "system": {"status": "healthy", "message": "System resources OK"},
                        "temp_directory": {"status": "healthy", "message": "Temp directory accessible"}
                    }
                }
            )
            
        status = await health_checker.get_status()
        return JSONResponse(
            status_code=200,
            content=status
        )
    except Exception as e:
        error_tracker.track_error(e, {"endpoint": "/health"})
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/ready")
async def readiness_check():
    """Check if the service is ready to handle requests."""
    try:
        status = await health_checker.get_status()
        is_ready = (
            status["checks"]["model"]["status"] == "ok" and
            status["checks"]["temp_directory"]["status"] == "ok"
        )
        return {
            "status": "ready" if is_ready else "not_ready",
            "checks": status
        }
    except Exception as e:
        error_tracker.track_error(e, {"endpoint": "/ready"})
        logger.error("readiness_check_failed", error=str(e))
        return {"status": "not_ready", "error": str(e)}

@app.get("/live")
async def liveness_check():
    """Simple liveness check."""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}

@app.post("/config")
async def update_config(new_config: ModelConfig):
    """Update model configuration."""
    global model, config
    try:
        logger.info(f"Updating configuration: {new_config.__dict__}")
        model = load_model(new_config)
        config = new_config
        return {"status": "success", "config": config.__dict__}
    except Exception as e:
        logger.error(f"Failed to update config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def process_audio(audio_path: str, options: Optional[TranscriptionOptions] = None) -> dict:
    """Process a single audio file."""
    try:
        opts = options.model_dump() if options else {}
        result = model.transcribe(audio_path, **opts)
        return {
            "text": result["text"],
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", [])
        }
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {str(e)}")
        raise e  # Raise the exception instead of returning error dict

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    options: Optional[TranscriptionOptions] = None
):
    """Transcribe audio file."""
    if not options:
        options = TranscriptionOptions()

    try:
        # Save uploaded file
        temp_file = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            # Validate audio file
            validate_audio_file(temp_file)
        except ValueError as e:
            cleanup_temp_file(temp_file)
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            cleanup_temp_file(temp_file)
            raise e

        # Process transcription in thread pool to not block
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, process_audio, temp_file, options)
        
        # Cleanup temp file
        cleanup_temp_file(temp_file)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        error_tracker.track_error(e, {"filename": file.filename})
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_transcribe")
async def batch_transcribe(request: BatchRequest):
    """Batch transcribe multiple audio files."""
    try:
        results = []
        loop = asyncio.get_event_loop()
        
        for file_path in request.files:
            # Validate each file
            validate_audio_file(file_path)
            
            # Process transcription
            result = await loop.run_in_executor(executor, process_audio, file_path, request.options)
            results.append({
                "file": file_path,
                "result": result
            })
        
        return {"results": results}
    except Exception as e:
        error_tracker.track_error(e, {"files": request.files})
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import sys
    import os
    
    # Add the project root directory to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    # Run the server
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    ) 
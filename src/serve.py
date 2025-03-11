import sys
import traceback
import psutil
import structlog
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import whisper
import torch
import os
import logging
from utils import validate_audio_file, cleanup_temp_file
import uuid
from typing import Optional, List, Dict
import time
from pydantic import BaseModel, Field, ConfigDict
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Ensure temp directory exists
TEMP_DIR = "/tmp/whisper_audio"
os.makedirs(TEMP_DIR, exist_ok=True)
logger.info(f"Using temp directory: {TEMP_DIR}")

# Prometheus metrics
REQUESTS = Counter("whisper_requests_total", "Total requests processed")
ERRORS = Counter("whisper_errors_total", "Total errors encountered", ["type"])
PROCESSING_TIME = Histogram("whisper_processing_seconds", "Time spent processing requests")
GPU_MEMORY = Gauge("whisper_gpu_memory_bytes", "GPU memory usage in bytes")
CPU_USAGE = Gauge("whisper_cpu_usage_percent", "CPU usage percentage")
MEMORY_USAGE = Gauge("whisper_memory_usage_bytes", "Memory usage in bytes")

# Model configuration
class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    whisper_model: str = Field("base", description="Whisper model size")
    device: str = Field("auto", description="Device to use")
    compute_type: str = Field("float32", description="Compute type")
    cpu_threads: int = Field(4, description="Number of CPU threads")
    num_workers: int = Field(2, description="Number of workers")

# Transcription options
class TranscriptionOptions(BaseModel):
    language: Optional[str] = Field(None, description="Language code (e.g., 'en', 'es')")
    task: str = Field("transcribe", description="Task type (transcribe or translate)")
    beam_size: int = Field(5, description="Beam size for beam search")
    patience: float = Field(1.0, description="Beam search patience factor")
    temperature: List[float] = Field([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], description="Temperature for sampling")

# Batch request
class BatchRequest(BaseModel):
    files: List[str] = Field(..., description="List of file paths to process")
    options: Optional[TranscriptionOptions] = None

# Global variables
config = ModelConfig()
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
    global model, config, executor
    try:
        logger.info("loading_model", config=config.model_dump())
        model = load_model(config)
        logger.info("model_loaded_successfully")
        
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

app = FastAPI(
    title="Whisper Transcription Service",
    description="Speech-to-text service using OpenAI's Whisper model",
    version="1.0.0",
    lifespan=lifespan
)

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
)

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
                    config=model_config.model_dump())
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed system information."""
    try:
        update_metrics()
        gpu_info = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
        
        process = psutil.Process()
        system_info = {
            "cpu_percent": process.cpu_percent(),
            "memory_percent": process.memory_percent(),
            "memory_info": dict(process.memory_info()._asdict())
        }
        
        return {
            "status": "healthy",
            "config": config.model_dump(),
            "gpu": gpu_info,
            "system": system_info,
            "model_loaded": model is not None
        }
    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config")
async def update_config(new_config: ModelConfig):
    """Update model configuration."""
    global model, config
    try:
        logger.info(f"Updating configuration: {new_config}")
        model = load_model(new_config)
        config = new_config
        return {"status": "success", "config": config.model_dump()}
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
        return {"error": str(e)}

@app.post("/batch")
async def batch_transcribe(
    request: BatchRequest,
    background_tasks: BackgroundTasks
) -> dict:
    """Batch transcription endpoint."""
    try:
        futures = []
        for file_path in request.files:
            if not os.path.exists(file_path):
                raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
            validate_audio_file(file_path)
            
        # Process files in parallel using ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor,
            lambda: {
                file_path: process_audio(file_path, request.options)
                for file_path in request.files
            }
        )
        
        return {
            "status": "success",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer")
async def infer(
    audio_file: UploadFile = File(...),
    options: Optional[Dict] = None
):
    """
    Transcribe audio file using Whisper.
    
    Args:
        audio_file (UploadFile): The audio file to transcribe
        options (Dict, optional): Transcription options
    
    Returns:
        dict: Transcription result
    """
    REQUESTS.inc()
    request_id = str(uuid.uuid4())
    log = logger.bind(request_id=request_id)
    
    if not audio_file:
        ERRORS.labels(type="no_file").inc()
        raise HTTPException(status_code=400, detail="No audio file provided")

    temp_file = f"{TEMP_DIR}/temp_audio_{request_id}"
    start_time = time.time()
    
    try:
        log.info("processing_started", filename=audio_file.filename)
        content = await audio_file.read()
        log.info("file_read", bytes_read=len(content))
        
        with open(temp_file, "wb") as f:
            f.write(content)
        log.info("file_saved", temp_file=temp_file)

        validate_audio_file(temp_file)
        log.info("file_validated")
        
        update_metrics()
        
        opts = options or {}
        log.info("starting_transcription", options=opts)
        
        with PROCESSING_TIME.time():
            result = model.transcribe(temp_file, **opts)
        
        processing_time = time.time() - start_time
        log.info("transcription_completed", 
                processing_time=processing_time,
                language=result.get("language"))
        
        response = {
            "text": result["text"],
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", []),
            "processing_time": f"{processing_time:.2f}s"
        }
        
        return JSONResponse(content=response)

    except Exception as e:
        processing_time = time.time() - start_time
        error_type = type(e).__name__
        ERRORS.labels(type=error_type).inc()
        
        log.error("transcription_failed",
                  error=str(e),
                  error_type=error_type,
                  traceback=traceback.format_exc(),
                  processing_time=processing_time)
        
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": error_type,
                "request_id": request_id
            }
        )
    finally:
        cleanup_temp_file(temp_file)
        update_metrics()

if __name__ == "__main__":
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except Exception as e:
        logger.error("server_startup_failed",
                    error=str(e),
                    traceback=traceback.format_exc())
        sys.exit(1) 
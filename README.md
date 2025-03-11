# kube-whisperer: Speech-to-Text Service with KServe

A speech-to-text service using OpenAI's Whisper model, deployed on Kubernetes with KServe and GPU acceleration.

## Quick Start

The easiest way to deploy the service is using our pre-built Docker image:

```bash
# Apply the InferenceService using the pre-built image (GPU version)
kubectl apply -f https://raw.githubusercontent.com/bnallapeta/kube-whisperer/main/k8s/whisper-prebuilt.yaml

# Or for CPU-only deployment
kubectl apply -f https://raw.githubusercontent.com/bnallapeta/kube-whisperer/main/k8s/whisper-prebuilt-cpu.yaml
```

Or create your own YAML file:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: whisper-service
  namespace: default
spec:
  predictor:
    containers:
    - image: ghcr.io/bnallapeta/kube-whisperer:0.0.1
      env:
      - name: WHISPER_MODEL
        value: "base"  # Options: tiny, base, small, medium, large
      - name: DEVICE
        value: "cuda"  # Use "cpu" for CPU-only deployment
      resources:
        limits:
          cpu: "4"
          memory: 8Gi
          nvidia.com/gpu: "1"  # Remove for CPU-only deployment
        requests:
          cpu: "1"
          memory: 4Gi
          nvidia.com/gpu: "1"  # Remove for CPU-only deployment
```

Once deployed, you can use the service:

```bash
# Test transcription
curl -X POST http://whisper-service.default.svc.cluster.local/transcribe \
  -F "file=@your-audio-file.wav"
```

For more detailed configuration and deployment options, see the [Configuration Options](#configuration-options) and [Deployment](#deployment) sections.

## Features

- 🎯 Uses OpenAI's Whisper model for accurate speech recognition (configurable: tiny, base, small, medium, large)
- 🚀 GPU-accelerated inference with NVIDIA GPUs (with CPU fallback)
- 🛡️ Deployed with KServe for production-grade serving
- 🔄 Automatic scaling with Knative
- 🌐 Accessible via domain routing
- ⚡ Fast inference times (~1s for 20s audio)
- 📊 Comprehensive monitoring with Prometheus metrics
- 🔍 Detailed health checks and diagnostics
- 🌍 Multi-language support with language selection
- 📦 Batch processing capabilities

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Building and Publishing](#building-and-publishing)
- [Deployment](#deployment)
- [Configuration Options](#configuration-options)
- [Testing the Service](#testing-the-service)
- [API Endpoints](#api-endpoints)
- [Performance](#performance)
- [Container Registry Setup](#container-registry-setup)
- [Production Considerations](#production-considerations)
- [Monitoring and Metrics](#monitoring-and-metrics)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Prerequisites

### Cloud Infrastructure
- Kubernetes cluster with:
  - KServe installed
  - NVIDIA GPU support
  - Istio for ingress
  - Knative for serverless deployment
  - Access to a container registry (ACR, Docker Hub, GCR, etc.)

### Local Development
- Python 3.11+
- Docker Desktop or Podman
- `kubectl` CLI
- `make` utility
- FFmpeg (`brew install ffmpeg` on macOS)

## Local Development

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/whisper
   cd whisper
   ```

2. **Set Up Python Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Run Local Server**
   ```bash
   # Start the development server
   make run-local

   # In another terminal, test the server
   make test-local AUDIO=samples/sample20s.wav
   ```

## Building and Publishing

### Using Makefile

The project includes a comprehensive Makefile for common tasks:

```bash
# Build container image (supports both Docker and Podman)
make build

# Run tests
make test

# Build and push to container registry
make push

# Deploy to Kubernetes
make deploy

# Clean up resources
make clean
```

### Manual Build Process

1. **Build Container Image**
   ```bash
   # Using Docker
   docker build -t whisper-service:latest .
   
   # Using Podman
   podman build -t whisper-service:latest .
   ```

2. **Tag for Container Registry**
   ```bash
   # For Azure Container Registry (ACR)
   docker tag whisper-service:latest <registry-name>.azurecr.io/whisper-service:latest
   # or with Podman
   podman tag whisper-service:latest <registry-name>.azurecr.io/whisper-service:latest

   # For Docker Hub
   docker tag whisper-service:latest <username>/whisper-service:latest
   # or with Podman
   podman tag whisper-service:latest <username>/whisper-service:latest
   ```

3. **Push to Container Registry**
   ```bash
   # For ACR
   docker push <registry-name>.azurecr.io/whisper-service:latest
   # or with Podman
   podman push <registry-name>.azurecr.io/whisper-service:latest

   # For Docker Hub
   docker push <username>/whisper-service:latest
   # or with Podman
   podman push <username>/whisper-service:latest
   ```

## Container Registry Setup

You can use any container registry that your Kubernetes cluster can access. Below are instructions for common options:

### Option 1: Azure Container Registry (ACR)

1. **Create ACR**
   ```bash
   az acr create --resource-group <resource-group> --name <registry-name> --sku Basic
   ```

2. **Create Image Pull Secret**
   ```bash
   # Using Azure CLI and Service Principal
   kubectl create secret docker-registry acr-secret \
     --docker-server=<registry-name>.azurecr.io \
     --docker-username=<service-principal-id> \
     --docker-password=<service-principal-password> \
     --namespace=default

   # Or using Azure CLI token (recommended for testing)
   kubectl create secret docker-registry acr-secret \
     --docker-server=<registry-name>.azurecr.io \
     --docker-username=00000000-0000-0000-0000-000000000000 \
     --docker-password=$(az acr login --name <registry-name> --expose-token --query accessToken -o tsv) \
     --namespace=default
   ```

3. **Login for Local Development**
   ```bash
   # Using Azure CLI
   az acr login --name <registry-name>

   # Using Podman with Service Principal
   podman login <registry-name>.azurecr.io -u <service-principal-id> -p <service-principal-password>
   ```

### Option 2: Docker Hub

1. **Create Image Pull Secret**
   ```bash
   kubectl create secret docker-registry dockerhub-secret \
     --docker-server=https://index.docker.io/v1/ \
     --docker-username=<your-username> \
     --docker-password=<your-password> \
     --docker-email=<your-email> \
     --namespace=default
   ```

2. **Login for Local Development**
   ```bash
   # Using Docker
   docker login

   # Using Podman
   podman login docker.io
   ```

### Option 3: Google Container Registry (GCR)

1. **Create Service Account and Key**
   ```bash
   # Create service account key
   gcloud iam service-accounts keys create key.json \
     --iam-account=<sa-name>@<project-id>.iam.gserviceaccount.com
   ```

2. **Create Image Pull Secret**
   ```bash
   kubectl create secret docker-registry gcr-secret \
     --docker-server=gcr.io \
     --docker-username=_json_key \
     --docker-password="$(cat key.json)" \
     --docker-email=<your-email> \
     --namespace=default
   ```

3. **Login for Local Development**
   ```bash
   # Using Docker
   gcloud auth configure-docker

   # Using Podman
   podman login gcr.io
   ```

## Configuration

### Environment Variables

Set these variables in your environment or Makefile:

```bash
# Container Registry Configuration
export REGISTRY_TYPE=acr                                  # or dockerhub, gcr
export REGISTRY_NAME=<registry-name>                      # e.g., myacr, docker.io/username
export REGISTRY_SECRET_NAME=acr-secret                    # Secret name created above
export REGISTRY_IMAGE=${REGISTRY_NAME}/whisper-service:latest

# For ACR
export ACR_SP_ID=<service-principal-id>                   # Optional: For service principal auth
export ACR_SP_PASSWORD=<service-principal-password>       # Optional: For service principal auth
```

### Makefile Configuration

The Makefile uses these variables to configure the build and deployment:

```makefile
REGISTRY_TYPE ?= acr
REGISTRY_NAME ?= <registry-name>
REGISTRY_SECRET_NAME ?= acr-secret
REGISTRY_IMAGE ?= $(REGISTRY_NAME)/whisper-service:latest

.PHONY: deploy
deploy:
    # Replace variables in yaml
    sed -e "s|\${REGISTRY_IMAGE}|$(REGISTRY_IMAGE)|g" \
        -e "s|\${REGISTRY_SECRET_NAME}|$(REGISTRY_SECRET_NAME)|g" \
        k8s/whisper-custom.yaml | kubectl apply -f -
```

## Configuration Options

The Whisper service can be configured in several ways:

### Model Configuration

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `whisper_model` | Whisper model size | `base` | `tiny`, `base`, `small`, `medium`, `large` |
| `device` | Computation device | `cpu` | `cpu`, `cuda`, `mps` |
| `compute_type` | Computation precision | `int8` | `int8`, `float16`, `float32` |
| `cpu_threads` | Number of CPU threads | `4` | Any positive integer |
| `num_workers` | Number of workers | `1` | Any positive integer |
| `download_root` | Model download directory | `/tmp/whisper_models` | Any valid directory path |

### Transcription Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `language` | Language code | `en` | Any ISO language code (e.g., `en`, `es`, `fr`) |
| `task` | Transcription task | `transcribe` | `transcribe`, `translate` |
| `beam_size` | Beam search size | `5` | Integer between 1-10 |
| `patience` | Beam search patience | `1.0` | Float between 0.0-2.0 |
| `temperature` | Sampling temperature | `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` | List of floats between 0.0-1.0 |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level | `INFO` |
| `TEMP_DIR` | Temporary directory for audio files | `/tmp/whisper_audio` |
| `MODEL_DOWNLOAD_ROOT` | Directory for model downloads | `/tmp/whisper_models` |

### How to Configure Parameters

You can configure these parameters in different ways depending on your deployment method:

#### 1. Runtime API Configuration

The most flexible way to configure the model is through the `/config` API endpoint:

```bash
# Update model configuration at runtime
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{
    "whisper_model": "small",
    "device": "cuda",
    "compute_type": "float16",
    "cpu_threads": 8,
    "num_workers": 2,
    "download_root": "/tmp/whisper_models"
  }'
```

#### 2. Per-Request Configuration (Transcription Options)

For transcription options, you can specify them per request:

```bash
# Single file transcription with options
curl -X POST http://localhost:8000/transcribe \
  -F "file=@samples/sample.wav" \
  -F "options={\"language\":\"es\",\"task\":\"translate\",\"beam_size\":8}"

# Batch transcription with options
curl -X POST http://localhost:8000/batch_transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "files": ["/path/to/file1.wav", "/path/to/file2.mp3"],
    "options": {
      "language": "fr",
      "task": "transcribe",
      "beam_size": 5,
      "patience": 1.0,
      "temperature": [0.0, 0.2, 0.4]
    }
  }'
```

#### 3. Environment Variables (Container/K8s Deployment)

When deploying with Kubernetes or Docker, set environment variables in your deployment:

```yaml
# In Kubernetes deployment (k8s/whisper-custom.yaml)
env:
  - name: WHISPER_MODEL
    value: "base"
  - name: DEVICE
    value: "cuda"
  - name: COMPUTE_TYPE
    value: "float16"
  - name: CPU_THREADS
    value: "4"
  - name: NUM_WORKERS
    value: "2"
  - name: MODEL_DOWNLOAD_ROOT
    value: "/tmp/whisper_models"
  - name: LOG_LEVEL
    value: "INFO"
```

```bash
# In Docker/Podman
docker run -e WHISPER_MODEL=medium \
           -e DEVICE=cuda \
           -e COMPUTE_TYPE=float16 \
           -e CPU_THREADS=8 \
           -p 8000:8000 \
           whisper-service:latest
```

#### 4. Direct Python Configuration (Local Development)

When running locally, you can modify `src/config.py` or set environment variables:

```python
# In your Python code
from src.config import ModelConfig

# Create custom configuration
config = ModelConfig(
    whisper_model="small",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16",
    cpu_threads=8,
    num_workers=2,
    download_root="/tmp/whisper_models"
)

# Use this config to load the model
model = load_model(config)
```

```bash
# Or set environment variables before running
export WHISPER_MODEL=small
export DEVICE=cuda
export COMPUTE_TYPE=float16
export CPU_THREADS=8
export NUM_WORKERS=2

# Then run the service
python -m src.serve
```

#### 5. Makefile Configuration (Build & Deployment)

When using the Makefile for deployment, you can set variables:

```bash
# Set variables for deployment
make deploy WHISPER_MODEL=medium DEVICE=cuda COMPUTE_TYPE=float16
```

### Configuration Precedence

The configuration parameters are applied with the following precedence (highest to lowest):

1. API endpoint configuration (`/config`)
2. Per-request options (for transcription options)
3. Environment variables
4. Default values in code

This allows you to set defaults at deployment time but override them as needed at runtime.

## Deployment

The service is deployed using KServe's InferenceService custom resource. You can customize the configuration by modifying the environment variables in the deployment YAML:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  annotations:
    sidecar.istio.io/inject: "false"
  name: kube-whisperer
  namespace: default
spec:
  predictor:
    containers:
    - env:
      # Model configuration
      - name: WHISPER_MODEL
        value: base  # Options: tiny, base, small, medium, large
      - name: DEVICE
        value: cuda  # Options: cpu, cuda, mps
      - name: COMPUTE_TYPE
        value: float16  # Options: int8, float16, float32
      - name: CPU_THREADS
        value: "4"  # Number of CPU threads to use
      - name: NUM_WORKERS
        value: "2"  # Number of worker processes
      
      # Service configuration
      - name: LOG_LEVEL
        value: "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
      - name: TEMP_DIR
        value: "/tmp/whisper_audio"
      - name: MODEL_DOWNLOAD_ROOT
        value: "/tmp/whisper_models"
      - name: MAX_FILE_SIZE
        value: "26214400"  # Max file size in bytes (25MB)
      
      image: <registry-name>.azurecr.io/whisper-service:latest
      resources:
        limits:
          cpu: "4"
          memory: 8Gi
          nvidia.com/gpu: "1"  # Remove for CPU-only deployment
        requests:
          cpu: "1"
          memory: 4Gi
          nvidia.com/gpu: "1"  # Remove for CPU-only deployment
```

### Deployment Methods

#### Method 1: Using kubectl directly

```bash
# Deploy with default configuration
kubectl apply -f k8s/whisper-custom.yaml

# Deploy with custom configuration using environment variables
# First, edit the YAML file to update environment variables
# Then apply the changes
kubectl apply -f k8s/whisper-custom.yaml
```

#### Method 2: Using the Makefile

```bash
# Deploy with default configuration
make deploy

# Deploy with custom configuration
make deploy WHISPER_MODEL=small DEVICE=cuda COMPUTE_TYPE=float16
```

#### Method 3: Using sed for dynamic configuration

```bash
# Deploy with custom configuration without editing the YAML file
sed -e "s|value: base|value: small|g" \
    -e "s|value: float16|value: int8|g" \
    -e "s|value: \"4\"|value: \"8\"|g" \
    k8s/whisper-custom.yaml | kubectl apply -f -
```

### Resource Requirements by Model Size

Different model sizes require different resources:

| Model Size | Min Memory | Recommended CPU | GPU Memory |
|------------|------------|-----------------|------------|
| tiny       | 1GB        | 2 cores         | 1GB        |
| base       | 2GB        | 4 cores         | 2GB        |
| small      | 4GB        | 4 cores         | 4GB        |
| medium     | 8GB        | 8 cores         | 8GB        |
| large      | 16GB       | 8+ cores        | 16GB       |

Adjust your deployment resources accordingly:

```yaml
resources:
  limits:
    cpu: "8"  # Increase for larger models
    memory: 16Gi  # Increase for larger models
    nvidia.com/gpu: "1"
  requests:
    cpu: "4"  # Increase for larger models
    memory: 8Gi  # Increase for larger models
    nvidia.com/gpu: "1"
```

### CPU-Only Deployment

For CPU-only deployment, remove the GPU resource requests:

```yaml
resources:
  limits:
    cpu: "8"
    memory: 16Gi
  requests:
    cpu: "4"
    memory: 8Gi
  # No GPU resources specified
```

And set the device to CPU:

```yaml
- name: DEVICE
  value: cpu
```

## Testing the Service

### Method 1: Port Forwarding (Local Testing)

```bash
# Start port forwarding
kubectl port-forward pod/kube-whisperer-predictor-00001-deployment-XXXXX 8000:8000

# Test health endpoint
curl http://localhost:8000/health

# Test transcription with default options
curl -X POST http://localhost:8000/transcribe \
  -F "file=@samples/sample20s.wav"

# Test transcription with custom options
curl -X POST http://localhost:8000/transcribe \
  -F "file=@samples/sample20s.wav" \
  -F 'options={"language":"es", "task":"translate", "beam_size":8}'

# Test batch transcription
curl -X POST http://localhost:8000/batch_transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "files": ["samples/sample1.wav", "samples/sample2.wav"],
    "options": {
      "language": "en",
      "task": "transcribe"
    }
  }'

# Update model configuration
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{
    "whisper_model": "small",
    "device": "cuda",
    "compute_type": "float16"
  }'
```

### Method 2: Domain Access (Production Method)

Using nip.io for domain routing:

```bash
# Replace <external-ip> with your cluster's external IP
SERVICE_URL="http://kube-whisperer.default.<external-ip>.nip.io"

# Test health endpoint
curl $SERVICE_URL/health

# Test transcription with default options
curl -X POST $SERVICE_URL/transcribe \
  -F "file=@samples/sample20s.wav"

# Test transcription with custom options
curl -X POST $SERVICE_URL/transcribe \
  -F "file=@samples/sample20s.wav" \
  -F 'options={"language":"fr", "task":"transcribe"}'
```

### Using the Makefile for Testing

The Makefile includes targets for testing:

```bash
# Test with a specific audio file
make test-local AUDIO=samples/sample20s.wav

# Test with custom options
make test-local AUDIO=samples/sample20s.wav ARGS="--language es --task translate"

# Test against deployed service
make cluster-test AUDIO=samples/sample20s.wav
```

### Testing Different Model Configurations

To test different model configurations:

1. **Update configuration via API**:
   ```bash
   # Switch to a smaller model for faster processing
   curl -X POST http://localhost:8000/config \
     -H "Content-Type: application/json" \
     -d '{"whisper_model": "tiny"}'
   
   # Switch to a larger model for better accuracy
   curl -X POST http://localhost:8000/config \
     -H "Content-Type: application/json" \
     -d '{"whisper_model": "medium"}'
   ```

2. **Restart service with different environment variables**:
   ```bash
   # Stop current service
   kubectl delete inferenceservice kube-whisperer
   
   # Deploy with different configuration
   sed -e 's/value: base/value: small/' \
       -e 's/value: float16/value: int8/' \
       k8s/whisper-custom.yaml | kubectl apply -f -
   ```

3. **Run locally with different configuration**:
   ```bash
   # Run with tiny model
   WHISPER_MODEL=tiny python -m src.serve
   
   # Run with medium model and GPU
   WHISPER_MODEL=medium DEVICE=cuda python -m src.serve
   ```

## API Endpoints

### Health and Status Endpoints

- **URL**: `/health`
- **Method**: GET
- **Description**: Comprehensive health check of the service
- **Response**: Service health status including model, GPU, system resources, and temp directory

- **URL**: `/ready`
- **Method**: GET
- **Description**: Readiness check for the service
- **Response**: Indicates if the service is ready to handle requests

- **URL**: `/live`
- **Method**: GET
- **Description**: Liveness check for the service
- **Response**: Simple response to indicate the service is running

### Configuration Endpoint

- **URL**: `/config`
- **Method**: POST
- **Content-Type**: application/json
- **Description**: Update model configuration
- **Request Body**:
  ```json
  {
    "whisper_model": "base",
    "device": "cpu",
    "compute_type": "int8",
    "cpu_threads": 4,
    "num_workers": 1,
    "download_root": "/tmp/whisper_models"
  }
  ```
- **Response**: Updated configuration

### Transcription Endpoints

- **URL**: `/transcribe`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Description**: Transcribe a single audio file
- **Parameters**: 
  - `file`: Audio file (WAV, MP3, M4A, etc.)
  - `options` (optional): JSON object with transcription options
- **Response**: Transcribed text with segments and metadata

- **URL**: `/batch_transcribe`
- **Method**: POST
- **Content-Type**: application/json
- **Description**: Batch transcribe multiple audio files
- **Request Body**:
  ```json
  {
    "files": ["/path/to/file1.wav", "/path/to/file2.mp3"],
    "options": {
      "language": "en",
      "task": "transcribe",
      "beam_size": 5,
      "patience": 1.0,
      "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    }
  }
  ```
- **Response**: Array of transcription results

### Metrics Endpoint

- **URL**: `/metrics`
- **Method**: GET
- **Description**: Prometheus metrics for monitoring
- **Response**: Prometheus-formatted metrics

## Performance

- Processing time: ~1s for 20s audio file (base model on GPU)
- GPU: NVIDIA GPUs significantly improve performance
- Model size impact:
  - tiny: Fastest, least accurate
  - base: Good balance of speed and accuracy
  - small: Better accuracy, slower than base
  - medium: High accuracy, requires more resources
  - large: Highest accuracy, requires significant resources
- Compute type impact:
  - int8: Fastest, slightly lower accuracy
  - float16: Good balance of speed and accuracy
- Processing time: ~1s for 20s audio file
- GPU: NVIDIA GPUs
- Model: Whisper base
- Compute type: float16

## Production Considerations

1. **Security**:
   - Use proper domain with TLS/HTTPS
   - Implement authentication
   - Configure network policies

2. **Scaling**:
   - Adjust resource requests/limits based on load
   - Configure Knative autoscaling parameters

3. **Monitoring**:
   - Add Prometheus metrics
   - Set up logging with EFK stack
   - Configure alerts for service health

## Monitoring and Metrics

The service exposes Prometheus metrics at the `/metrics` endpoint, including:

- `whisper_requests_total`: Total number of requests processed
- `whisper_errors_total`: Total number of errors by type
- `whisper_processing_seconds`: Histogram of processing times
- `whisper_gpu_memory_bytes`: GPU memory usage
- `whisper_cpu_usage_percent`: CPU usage percentage
- `whisper_memory_usage_bytes`: Memory usage

### [TODO] Grafana Dashboard

A sample Grafana dashboard is available in the `monitoring/` directory. To import it:

1. Access your Grafana instance
2. Navigate to Dashboards > Import
3. Upload the JSON file or paste its contents
4. Select your Prometheus data source
5. Click Import

### [TODO] Alerting

Sample Prometheus alerting rules are provided in `monitoring/alerts.yaml`. These include:

- High error rate alerts
- Service availability alerts
- Resource utilization alerts

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure sufficient memory for the selected model
   - Check model download directory permissions
   - Try a smaller model size

2. **GPU-related Issues**
   - Verify GPU drivers are installed and working
   - Check CUDA compatibility
   - Fall back to CPU if GPU issues persist

3. **Audio Processing Errors**
   - Ensure audio file is in a supported format
   - Check file permissions
   - Verify file is not corrupted

4. **Performance Issues**
   - Adjust model size based on available resources
   - Increase CPU threads for CPU-based inference
   - Use batch processing for multiple files

### Logs and Diagnostics

- Check service logs for detailed error information
- Use the `/health` endpoint for comprehensive diagnostics
- Monitor metrics for performance bottlenecks

### Cleanup

If you encounter disk space issues or need to clean up build artifacts:

```bash
# Clean up build artifacts
./clean.sh

# Or use make
make clean-artifacts
```

## Contributing

### Development Workflow

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/whisper
   cd whisper
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Write code
   - Add tests
   - Update documentation

4. **Test Locally**
   ```bash
   make test-local
   ```

5. **Submit PR**
   - Create pull request
   - Add description of changes
   - Link related issues

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for functions
- Keep functions small and focused

### Testing
- Write unit tests for new features
- Ensure all tests pass locally
- Add integration tests for API endpoints

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

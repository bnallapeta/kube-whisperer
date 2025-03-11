# Whisper Speech-to-Text Service with KServe

A speech-to-text service using OpenAI's Whisper model, deployed on Kubernetes with KServe and GPU acceleration.

## Features

- 🎯 Uses OpenAI's Whisper base model for accurate speech recognition
- 🚀 GPU-accelerated inference with NVIDIA GPUs
- 🛡️ Deployed with KServe for production-grade serving
- 🔄 Automatic scaling with Knative
- 🌐 Accessible via domain routing
- ⚡ Fast inference times (~1s for 20s audio)

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Building and Publishing](#building-and-publishing)
- [Deployment](#deployment)
- [Testing the Service](#testing-the-service)
- [API Endpoints](#api-endpoints)
- [Performance](#performance)
- [Container Registry Setup](#container-registry-setup)
- [Production Considerations](#production-considerations)
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

## Deployment

The service is deployed using KServe's InferenceService custom resource:

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
      - name: WHISPER_MODEL
        value: base
      - name: COMPUTE_TYPE
        value: float16
      - name: CPU_THREADS
        value: "4"
      - name: NUM_WORKERS
        value: "2"
      - name: MAX_FILE_SIZE
        value: "26214400"
      image: <registry-name>.azurecr.io/whisper-service:latest
      resources:
        limits:
          cpu: "4"
          memory: 8Gi
          nvidia.com/gpu: "1"
        requests:
          cpu: "1"
          memory: 4Gi
          nvidia.com/gpu: "1"
```

Deploy using:
```bash
kubectl apply -f k8s/whisper-custom.yaml
```

## Testing the Service

### Method 1: Port Forwarding (Local Testing)

```bash
# Start port forwarding
kubectl port-forward pod/kube-whisperer-predictor-00001-deployment-XXXXX 8000:8000

# Test health endpoint
curl http://localhost:8000/health

# Test transcription
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@samples/sample20s.wav"
```

### Method 2: Domain Access (Production Method)

Using nip.io for domain routing:

```bash
# Replace <external-ip> with your cluster's external IP
# Test health endpoint
curl http://kube-whisperer.default.<external-ip>.nip.io/health

# Test transcription
curl -X POST http://kube-whisperer.default.<external-ip>.nip.io/infer \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@samples/sample20s.wav"
```

## API Endpoints

### Health Check
- **URL**: `/health`
- **Method**: GET
- **Response**: Service health status and configuration

### Transcription
- **URL**: `/infer`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Parameters**: 
  - `audio_file`: Audio file (WAV, MP3, etc.)
- **Response**: Transcribed text with timestamps and metadata

## Performance

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

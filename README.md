# kube-whisperer: Speech-to-Text Service

> Fast, reliable speech recognition using OpenAI's Whisper models, optimized for Kubernetes deployments.

[![KServe](https://img.shields.io/badge/KServe-Ready-blue.svg)]()
[![GPU](https://img.shields.io/badge/GPU-Accelerated-green.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)]()

## Table of Contents
- [Features](#features)
- [Deployment Options](#deployment-options)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## Features

- **Multiple Model Sizes**: Choose from 5 model sizes (tiny to large) balancing speed vs. accuracy
- **GPU Acceleration**: NVIDIA GPU support for faster processing
- **Multi-language**: Supports 90+ languages with language auto-detection
- **Translation**: Translates non-English speech to English
- **Production-Ready**: Built for Kubernetes with monitoring, health checks, and scaling
- **Flexible Configuration**: Runtime configuration via API or environment variables

## Deployment Options

### Kubernetes with KServe

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: whisper-service
spec:
  predictor:
    containers:
    - image: ghcr.io/bnallapeta/kube-whisperer:0.0.1
      env:
      - name: WHISPER_MODEL
        value: "base"  # Options: tiny, base, small, medium, large
      - name: DEVICE
        value: "cuda"  # Use "cpu" for CPU-only
      resources:
        limits:
          nvidia.com/gpu: "1"  # Remove for CPU deployment
```

### Docker / Podman

```bash
docker run -p 8000:8000 \
  -e WHISPER_MODEL=small \
  -e DEVICE=cuda \
  -e COMPUTE_TYPE=float16 \
  ghcr.io/bnallapeta/kube-whisperer:0.0.1
```

### Local Development

```bash
# Clone repository
git clone https://github.com/bnallapeta/kube-whisperer
cd kube-whisperer

# activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run locally with configuration
WHISPER_MODEL=medium DEVICE=cpu python -m src.serve
```

## Configuration

### Core Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `whisper_model` | Model size | `base` | `tiny`, `base`, `small`, `medium`, `large` |
| `device` | Compute device | `cpu` | `cpu`, `cuda`, `mps` |
| `compute_type` | Precision | `int8` | `int8`, `float16`, `float32` |
| `cpu_threads` | CPU threads | `4` | Any positive integer |
| `num_workers` | Workers | `1` | Any positive integer |

### Transcription Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `language` | Language code | `en` | Any ISO language code (e.g., `en`, `es`, `fr`) |
| `task` | Task type | `transcribe` | `transcribe`, `translate` |
| `beam_size` | Beam search size | `5` | Integer between 1-10 |
| `temperature` | Sampling temperature | `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` | Float values 0.0-1.0 |

### Configuration Methods

#### 1. Kubernetes Deployment

For Kubernetes deployments, configuration values are specified directly in the YAML file:

```yaml
# In your InferenceService YAML
spec:
  predictor:
    containers:
    - image: ghcr.io/bnallapeta/kube-whisperer:0.0.1
      env:
      - name: WHISPER_MODEL
        value: "small"
      - name: DEVICE
        value: "cuda"
      - name: COMPUTE_TYPE
        value: "float16"
      - name: CPU_THREADS
        value: "8"
      - name: NUM_WORKERS
        value: "2"
```

To modify these configurations:
- Edit the YAML file before applying
- Use `kubectl edit inferenceservice whisper-service` to modify an existing deployment
- Use Kustomize or Helm for managing different configurations

#### 2. Docker Environment Variables

When using Docker, pass environment variables with the `-e` flag:

```bash
docker run -p 8000:8000 \
  -e WHISPER_MODEL=small \
  -e DEVICE=cuda \
  -e COMPUTE_TYPE=float16 \
  -e CPU_THREADS=8 \
  ghcr.io/bnallapeta/kube-whisperer:0.0.1
```

#### 3. Local Development Environment Variables

For local development, set environment variables before running:

```bash
# Set environment variables
export WHISPER_MODEL=small
export DEVICE=cuda
export COMPUTE_TYPE=float16

# Then run the service
python -m src.serve
```

#### 4. Runtime API Configuration

Update configuration without restarting:

```bash
curl -X POST http://whisper-service/config \
  -H "Content-Type: application/json" \
  -d '{
    "whisper_model": "small",
    "device": "cuda",
    "compute_type": "float16"
  }'
```

#### 5. Per-Request Options

Configure options for a specific request:

```bash
curl -X POST http://whisper-service/transcribe \
  -F "file=@audio.wav" \
  -F 'options={"language":"es", "task":"translate"}'
```

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/transcribe` | POST | Transcribe a single audio file |
| `/batch_transcribe` | POST | Transcribe multiple audio files |
| `/config` | POST | Update service configuration |
| `/ready` | GET | Readiness check |
| `/live` | GET | Liveness check |
| `/metrics` | GET | Prometheus metrics |

### Examples

#### Transcribe Audio

```bash
curl -X POST http://whisper-service/transcribe \
  -F "file=@audio.wav"
```

#### Transcribe with Options

```bash
curl -X POST http://whisper-service/transcribe \
  -F "file=@audio.wav" \
  -F 'options={"language":"fr", "task":"translate"}'
```

#### Batch Transcription

```bash
curl -X POST http://whisper-service/batch_transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "files": ["/path/to/file1.wav", "/path/to/file2.mp3"],
    "options": {"language": "es"}
  }'
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Choose a smaller model or increase GPU memory
   ```bash
   # Use a smaller model
   -e WHISPER_MODEL=small
   
   # Use lower precision
   -e COMPUTE_TYPE=int8
   ```

2. **Slow Performance**: Check your compute configuration
   ```bash
   # Check if GPU is being used
   curl http://whisper-service/ready | grep device
   
   # For CPU deployment, increase threads
   -e CPU_THREADS=8
   ```

3. **Language Issues**: Specify the language explicitly
   ```bash
   # Specify language in the request
   -F 'options={"language":"fr"}'
   ```

### Getting Help

- Check logs: `kubectl logs -f deployment/whisper-service`
- File issues: [GitHub Issues](https://github.com/bnallapeta/kube-whisperer/issues)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

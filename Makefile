# Phony targets declaration
.PHONY: all build push deploy test run clean install install-test help \
        cluster-deploy cluster-test registry-start registry-stop dev-build \
        dev-push local-deploy cloud-deploy setup-local run-local test-local \
        debug-deps debug-container clean-local create-secret show-config venv \
        cache-clean acr-login acr-build acr-push acr-clean acr-rebuild check-env \
        clean-artifacts

# Core variables
REGISTRY_TYPE ?= acr
REGISTRY_NAME ?= ${ACR_NAME}
REGISTRY_URL ?= $(if $(filter acr,$(REGISTRY_TYPE)),$(REGISTRY_NAME).azurecr.io,\
                $(if $(filter ghcr,$(REGISTRY_TYPE)),ghcr.io/${GITHUB_USERNAME},\
                $(if $(filter dockerhub,$(REGISTRY_TYPE)),docker.io/${DOCKER_USERNAME},\
                $(REGISTRY_NAME))))

# Image configuration
IMAGE_NAME ?= whisper-service
TAG ?= 0.0.3
REGISTRY_IMAGE = $(REGISTRY_URL)/$(IMAGE_NAME):$(TAG)
LOCAL_REGISTRY ?= localhost:5000
LOCAL_IMAGE_NAME = $(LOCAL_REGISTRY)/$(IMAGE_NAME):$(TAG)
REGISTRY_SECRET_NAME ?= acr-secret

# Build configuration
CONTAINER_RUNTIME ?= $(shell which podman 2>/dev/null || which docker 2>/dev/null)
PLATFORMS ?= linux/amd64,linux/arm64
CACHE_DIR ?= $(HOME)/.cache/whisper-build
BUILD_JOBS ?= 2

# Development configuration
PYTHON ?= python3
VENV ?= venv
PIP ?= $(VENV)/bin/pip
KUBECONFIG ?= ${KUBECONFIG}
PORT ?= 8000

# Create cache directories
$(shell mkdir -p $(CACHE_DIR)/amd64 $(CACHE_DIR)/arm64)

# Default target
all: build

# Virtual environment setup
venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Local development setup
setup-local:
	@echo "Setting up local development environment..."
	mkdir -p /tmp/whisper_audio
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

# Local registry commands
registry-start:
	@echo "Starting local registry..."
	-$(CONTAINER_RUNTIME) rm -f local-registry 2>/dev/null || true
	$(CONTAINER_RUNTIME) run -d --name local-registry -p 5000:5000 registry:2

registry-stop:
	@echo "Stopping local registry..."
	-$(CONTAINER_RUNTIME) stop local-registry
	-$(CONTAINER_RUNTIME) rm local-registry

# Build and push commands
build:
	$(CONTAINER_RUNTIME) build -t $(IMAGE_NAME):$(TAG) .
	$(CONTAINER_RUNTIME) tag $(IMAGE_NAME):$(TAG) $(REGISTRY_IMAGE)

push: check-env
	$(CONTAINER_RUNTIME) push $(REGISTRY_IMAGE)

# Multi-arch build commands
acr-login:
	@echo "Checking registry login status..."
	@if ! $(CONTAINER_RUNTIME) login $(REGISTRY_URL) >/dev/null 2>&1; then \
		echo "Not logged into registry. Please login first."; \
		exit 1; \
	fi

acr-build-only:
	@echo "Starting multi-architecture build..."
	@echo "Using container runtime: $(CONTAINER_RUNTIME)"
	@if echo $(CONTAINER_RUNTIME) | grep -q "docker"; then \
		echo "Using Docker for multi-architecture build..."; \
		$(CONTAINER_RUNTIME) buildx create --name multiarch --use || true; \
		$(CONTAINER_RUNTIME) buildx build \
			--platform $(PLATFORMS) \
			--cache-from type=local,src=$(CACHE_DIR) \
			--cache-to type=local,dest=$(CACHE_DIR) \
			--tag $(REGISTRY_IMAGE) \
			--load .; \
		$(CONTAINER_RUNTIME) buildx rm multiarch; \
	else \
		echo "Using Podman for multi-architecture build..."; \
		echo "Building AMD64 image..."; \
		$(CONTAINER_RUNTIME) build \
			--platform=linux/amd64 \
			--tag $(REGISTRY_IMAGE)-amd64 \
			--layers \
			--force-rm=false . || exit 1; \
		echo "Building ARM64 image..."; \
		$(CONTAINER_RUNTIME) build \
			--platform=linux/arm64 \
			--tag $(REGISTRY_IMAGE)-arm64 \
			--layers \
			--force-rm=false . || exit 1; \
	fi

acr-push:
	@if echo $(CONTAINER_RUNTIME) | grep -q "docker"; then \
		$(CONTAINER_RUNTIME) push $(REGISTRY_IMAGE); \
	else \
		$(CONTAINER_RUNTIME) push $(REGISTRY_IMAGE)-amd64 || exit 1; \
		$(CONTAINER_RUNTIME) push $(REGISTRY_IMAGE)-arm64 || exit 1; \
		$(CONTAINER_RUNTIME) manifest create $(REGISTRY_IMAGE) --amend || exit 1; \
		$(CONTAINER_RUNTIME) manifest add $(REGISTRY_IMAGE) $(REGISTRY_IMAGE)-amd64 || exit 1; \
		$(CONTAINER_RUNTIME) manifest add $(REGISTRY_IMAGE) $(REGISTRY_IMAGE)-arm64 || exit 1; \
		$(CONTAINER_RUNTIME) manifest push $(REGISTRY_IMAGE) || exit 1; \
	fi

acr-build: check-env acr-build-only acr-push
acr-rebuild: acr-clean acr-build

# Deployment commands
deploy: check-env
	sed -e "s|\$${REGISTRY_IMAGE}|$(REGISTRY_IMAGE)|g" \
	    -e "s|\$${REGISTRY_SECRET_NAME}|$(REGISTRY_SECRET_NAME)|g" \
	    k8s/whisper-custom.yaml | kubectl apply -f -

local-deploy:
	sed 's|image:.*|image: $(LOCAL_IMAGE_NAME)|' k8s/whisper-custom.yaml | kubectl apply -f -

cloud-deploy: check-env
	@echo "Deploying to cloud using $(REGISTRY_IMAGE)..."
	sed 's|image:.*|image: $(REGISTRY_IMAGE)|' k8s/whisper-custom.yaml | kubectl apply -f -

# Testing commands
run:
	$(PYTHON) src/serve.py

run-local: setup-local
	@echo "Starting Whisper service locally..."
	. $(VENV)/bin/activate && PORT=$(PORT) $(PYTHON) -m src.serve

test-local: setup-local
	@if [ -z "$(AUDIO)" ]; then \
		echo "Error: AUDIO file path not specified"; \
		echo "Usage: make test-local AUDIO=path/to/audio.wav"; \
		exit 1; \
	fi
	@echo "Testing local service with $(AUDIO)..."
	. $(VENV)/bin/activate && $(PYTHON) tests/api/test_transcription.py --url http://localhost:8000 --audio $(AUDIO) $(ARGS)

cluster-test:
	kubectl port-forward svc/kube-whisperer-predictor-default 8000:8000 & \
	sleep 5 && \
	$(PYTHON) tests/api/test_transcription.py --url http://localhost:8000 --audio $(AUDIO) $(ARGS)

# Cleanup commands
clean: clean-artifacts clean-local
	@echo "Clean complete!"

clean-local:
	rm -rf $(VENV)
	rm -rf /tmp/whisper_audio
	rm -rf __pycache__
	rm -rf *.pyc
	rm -rf .pytest_cache

acr-clean:
	@echo "Cleaning up registry images and manifests..."
	-$(CONTAINER_RUNTIME) manifest rm $(REGISTRY_IMAGE) 2>/dev/null || true
	-$(CONTAINER_RUNTIME) rmi $(REGISTRY_IMAGE)-amd64 2>/dev/null || true
	-$(CONTAINER_RUNTIME) rmi $(REGISTRY_IMAGE)-arm64 2>/dev/null || true
	-$(CONTAINER_RUNTIME) rmi $(REGISTRY_IMAGE) 2>/dev/null || true

cache-clean:
	rm -rf $(CACHE_DIR)/*

# Debug commands
debug-deps:
	@echo "Checking dependencies..."
	@. $(VENV)/bin/activate && $(PYTHON) -c "import sys; print(f'Python version: {sys.version}')"
	@. $(VENV)/bin/activate && $(PYTHON) -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
	@. $(VENV)/bin/activate && $(PYTHON) -c "import whisper; print(f'Whisper version: {whisper.__version__}')"
	@. $(VENV)/bin/activate && pip list

debug-container:
	@echo "Building and running container for debugging..."
	$(CONTAINER_RUNTIME) build -t whisper-debug .
	$(CONTAINER_RUNTIME) run -it --entrypoint=/bin/bash whisper-debug

# Registry secret creation
create-secret:
ifeq ($(REGISTRY_TYPE),acr)
	kubectl create secret docker-registry $(REGISTRY_SECRET_NAME) \
		--docker-server=$(REGISTRY_NAME).azurecr.io \
		--docker-username=00000000-0000-0000-0000-000000000000 \
		--docker-password=$$(az acr login --name $(REGISTRY_NAME) --expose-token --query accessToken -o tsv) \
		--namespace=default
else ifeq ($(REGISTRY_TYPE),dockerhub)
	@echo "Please create Docker Hub secret manually using:"
	@echo "kubectl create secret docker-registry $(REGISTRY_SECRET_NAME) \\"
	@echo "  --docker-server=https://index.docker.io/v1/ \\"
	@echo "  --docker-username=<your-username> \\"
	@echo "  --docker-password=<your-password> \\"
	@echo "  --docker-email=<your-email> \\"
	@echo "  --namespace=default"
else ifeq ($(REGISTRY_TYPE),gcr)
	@echo "Please create GCR secret manually using:"
	@echo "kubectl create secret docker-registry $(REGISTRY_SECRET_NAME) \\"
	@echo "  --docker-server=gcr.io \\"
	@echo "  --docker-username=_json_key \\"
	@echo "  --docker-password=\"\$$(cat key.json)\" \\"
	@echo "  --docker-email=<your-email> \\"
	@echo "  --namespace=default"
endif

# Environment checks
check-env:
	@if [ -z "$(REGISTRY_TYPE)" ]; then \
		echo "Error: REGISTRY_TYPE is not set"; \
		echo "Please set your registry type:"; \
		echo "  export REGISTRY_TYPE=acr|ghcr|dockerhub"; \
		exit 1; \
	fi
ifeq ($(REGISTRY_TYPE),acr)
	@if [ -z "$(ACR_NAME)" ]; then \
		echo "Error: ACR_NAME is not set"; \
		echo "Please set your Azure Container Registry name:"; \
		echo "  export ACR_NAME=your-acr-name"; \
		exit 1; \
	fi
else ifeq ($(REGISTRY_TYPE),ghcr)
	@if [ -z "$(GITHUB_USERNAME)" ]; then \
		echo "Error: GITHUB_USERNAME is not set"; \
		echo "Please set your GitHub username:"; \
		echo "  export GITHUB_USERNAME=your-username"; \
		exit 1; \
	fi
else ifeq ($(REGISTRY_TYPE),dockerhub)
	@if [ -z "$(DOCKER_USERNAME)" ]; then \
		echo "Error: DOCKER_USERNAME is not set"; \
		echo "Please set your Docker Hub username:"; \
		echo "  export DOCKER_USERNAME=your-username"; \
		exit 1; \
	fi
endif
	@if [ -z "$(KUBECONFIG)" ]; then \
		echo "Error: KUBECONFIG is not set"; \
		echo "Please set your kubeconfig path:"; \
		echo "  export KUBECONFIG=/path/to/your/kubeconfig"; \
		exit 1; \
	fi

# Configuration display
show-config:
	@echo "Current Configuration:"
	@echo "  REGISTRY_TYPE:     $(REGISTRY_TYPE)"
	@echo "  REGISTRY_URL:      $(REGISTRY_URL)"
	@echo "  REGISTRY_IMAGE:    $(REGISTRY_IMAGE)"
	@echo "  IMAGE_NAME:        $(IMAGE_NAME)"
	@echo "  TAG:               $(TAG)"
	@echo "  CONTAINER_RUNTIME: $(CONTAINER_RUNTIME)"
	@echo "  KUBECONFIG:        $(KUBECONFIG)"
ifeq ($(REGISTRY_TYPE),acr)
	@echo "  ACR_NAME:          $(ACR_NAME)"
else ifeq ($(REGISTRY_TYPE),ghcr)
	@echo "  GITHUB_USERNAME:   $(GITHUB_USERNAME)"
else ifeq ($(REGISTRY_TYPE),dockerhub)
	@echo "  DOCKER_USERNAME:   $(DOCKER_USERNAME)"
endif

# Help
help:
	@echo "Available commands:"
	@echo "  Local Development:"
	@echo "    make registry-start  - Start local registry"
	@echo "    make registry-stop   - Stop local registry"
	@echo "    make build          - Build image for local use"
	@echo "    make push           - Push image to registry"
	@echo "    make local-deploy   - Deploy to local cluster"
	@echo "    make run-local      - Run service locally"
	@echo ""
	@echo "  Cloud Development:"
	@echo "    make acr-build      - Build multi-arch image"
	@echo "    make cloud-deploy   - Deploy to cloud cluster"
	@echo ""
	@echo "  Testing:"
	@echo "    make cluster-test  - Test deployment on cluster (requires AUDIO=path/to/audio.wav)"
	@echo "    make test-local    - Test local service (requires AUDIO=path/to/audio.wav)"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make clean         - Clean up all resources"
	@echo "    make debug-deps    - Check dependencies"
	@echo "    make show-config   - Show current configuration"
	@echo ""
	@echo "Required Environment Variables:"
	@echo "  REGISTRY_TYPE      - Registry type (acr, ghcr, or dockerhub)"
	@echo "  KUBECONFIG        - Path to your kubeconfig file"
	@echo ""
	@echo "Registry-Specific Variables (based on REGISTRY_TYPE):"
	@echo "  For ACR (REGISTRY_TYPE=acr):"
	@echo "    ACR_NAME        - Your Azure Container Registry name"
	@echo ""
	@echo "  For GitHub (REGISTRY_TYPE=ghcr):"
	@echo "    GITHUB_USERNAME - Your GitHub username"
	@echo ""
	@echo "  For Docker Hub (REGISTRY_TYPE=dockerhub):"
	@echo "    DOCKER_USERNAME - Your Docker Hub username"
	@echo ""
	@echo "Optional Environment Variables:"
	@echo "  CONTAINER_RUNTIME - Container runtime to use (default: auto-detected podman or docker)"
	@echo "  LOCAL_REGISTRY    - Local registry address (default: localhost:5000)"
	@echo "  IMAGE_NAME       - Image name (default: whisper-service)"
	@echo "  TAG             - Image tag (default: latest)"

# Clean build artifacts
clean-artifacts:
	@echo "Cleaning build artifacts..."
	@./utils/clean.sh

# Clean all (including Docker images)
clean: clean-artifacts clean-local
	@echo "Clean complete!" 

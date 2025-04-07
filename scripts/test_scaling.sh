#!/bin/bash

# Configuration
SERVICE_URL="http://kube-whisperer.default.74.224.102.71.nip.io"
REQUESTS_PER_BATCH=10
REQUESTS_PER_SECOND=0.5  # Changed from 1 to 0.5 (1 request every 2 seconds)
MONITOR_INTERVAL=5
DESIRED_PODS=2
MAX_WAIT_TIME=180  # Maximum time to wait for pods to be ready (seconds)
AUDIO_FILE="samples/test.wav"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print a section header
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

# Function to print a success message
print_success() {
    echo -e "${GREEN}$1${NC}"
}

# Function to print a waiting message
print_waiting() {
    echo -e "${YELLOW}$1${NC}"
}

# Function to check pod count and details
check_pods() {
    export KUBECONFIG=/Users/bnr/work/openstack/kubeconfigs/s-azure-standalone.kubeconfig
    echo -e "${BLUE}=== Current Pods ===${NC}"
    kubectl get pods -l serving.kserve.io/inferenceservice=kube-whisperer -o wide
    echo -e "${BLUE}=== Pod Count ===${NC}"
    kubectl get pods -l serving.kserve.io/inferenceservice=kube-whisperer --no-headers | wc -l | tr -d ' '
}

# Function to analyze pod logs and show request distribution
analyze_request_distribution() {
    print_header "REQUEST DISTRIBUTION ANALYSIS"
    echo "Analyzing which pod handled which request..."
    
    export KUBECONFIG=/Users/bnr/work/openstack/kubeconfigs/s-azure-standalone.kubeconfig
    
    # Get pod names
    pod1=$(kubectl get pods -l serving.kserve.io/inferenceservice=kube-whisperer -o jsonpath='{.items[0].metadata.name}')
    pod2=$(kubectl get pods -l serving.kserve.io/inferenceservice=kube-whisperer -o jsonpath='{.items[1].metadata.name}')
    
    echo -e "${CYAN}Pod 1: $pod1${NC}"
    echo -e "${CYAN}Pod 2: $pod2${NC}"
    echo
    
    # Get the count of successful requests by counting 200 OK responses
    pod1_success=$(kubectl logs $pod1 --tail=200 | grep "POST /transcribe" | grep "200 OK" | wc -l | tr -d ' ')
    pod2_success=$(kubectl logs $pod2 --tail=200 | grep "POST /transcribe" | grep "200 OK" | wc -l | tr -d ' ')
    
    # Get the count of failed requests (500 errors)
    pod1_errors=$(kubectl logs $pod1 --tail=200 | grep "POST /transcribe" | grep "500 Internal Server Error" | wc -l | tr -d ' ')
    pod2_errors=$(kubectl logs $pod2 --tail=200 | grep "POST /transcribe" | grep "500 Internal Server Error" | wc -l | tr -d ' ')
    
    # Calculate total requests
    pod1_total=$((pod1_success + pod1_errors))
    pod2_total=$((pod2_success + pod2_errors))
    total_requests=$((pod1_total + pod2_total))
    
    echo -e "${GREEN}Request Distribution:${NC}"
    echo -e "Pod 1 handled ${CYAN}$pod1_total${NC} requests"
    echo -e "Pod 2 handled ${CYAN}$pod2_total${NC} requests"
    
    # Calculate distribution percentage
    if [ $total_requests -gt 0 ]; then
        pod1_percent=$((pod1_total * 100 / total_requests))
        pod2_percent=$((pod2_total * 100 / total_requests))
        echo -e "Distribution: ${CYAN}$pod1_percent%${NC} / ${CYAN}$pod2_percent%${NC}"
    fi
    
    echo
}

# Function to send a request
send_request() {
    local id=$1
    
    echo "Sending request $id..."
    curl -s -X POST "$SERVICE_URL/transcribe" \
         -H "Content-Type: multipart/form-data" \
         -F "file=@/Users/bnr/work/github/whisper/samples/test.wav" > /dev/null
    
    if [ $? -eq 0 ]; then
        echo "Request $id completed"
        return 0
    else
        echo "Request $id failed"
        return 1
    fi
}

# Function to send a batch of requests
send_batch() {
    local batch_num=$1
    local start_id=$2
    local count=$3
    
    print_header "PHASE $batch_num: SENDING REQUESTS"
    echo "Sending $count requests starting from ID $start_id"
    
    for i in $(seq $start_id $((start_id + count - 1))); do
        send_request $i &
        sleep 2  # Fixed 2-second delay between requests
    done
    
    # Wait for all requests in this batch to complete
    wait
    print_success "Batch $batch_num complete"
}

# Function to display a progress bar
show_progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((width * current / total))
    local remaining=$((width - completed))
    
    printf "\r["
    printf "%${completed}s" | tr " " "="
    printf "%${remaining}s" | tr " " " "
    printf "] %3d%%" $percentage
}

echo -e "${BLUE}=== KSERVE AUTO-SCALING DEMO FOR WHISPER ===${NC}"
echo "This demo will show how KServe automatically scales based on load"
echo "and distributes requests across multiple pods."
echo

print_header "INITIAL STATE"
echo "Initial pod count:"
check_pods
echo

# Phase 1: Send first batch to trigger scaling
print_header "PHASE 1: TRIGGERING SCALE-UP"
echo "Sending first batch of requests to trigger pod scaling..."
send_batch 1 1 $REQUESTS_PER_BATCH

# Wait for the second pod to be ready
print_header "WAITING FOR SCALING"
echo "Waiting for $DESIRED_PODS pods to be ready..."
start_time=$(date +%s)
pods_ready=false

while [ "$pods_ready" = false ]; do
    # Check if we've exceeded the maximum wait time
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    if [ $elapsed -gt $MAX_WAIT_TIME ]; then
        echo "Timeout waiting for pods to be ready after $MAX_WAIT_TIME seconds"
        exit 1
    fi
    
    # Get detailed pod information
    export KUBECONFIG=/Users/bnr/work/openstack/kubeconfigs/s-azure-standalone.kubeconfig
    
    # Get the total number of pods
    total_pods=$(kubectl get pods -l serving.kserve.io/inferenceservice=kube-whisperer --no-headers | wc -l | tr -d ' ')
    
    # Get container readiness for all pods in service
    pod_status=$(kubectl get pods -l serving.kserve.io/inferenceservice=kube-whisperer -o jsonpath='{range .items[*]}{range .status.containerStatuses[*]}{.ready}{" "}{end}{"\n"}{end}')
    
    # Count number of 'true' statuses
    ready_count=$(echo "$pod_status" | grep -o 'true' | wc -l)
    container_count=$(echo "$pod_status" | wc -w)
    
    # Show progress bar
    show_progress $elapsed $MAX_WAIT_TIME
    echo -n " - Pods: $total_pods, Ready containers: $ready_count/$container_count"
    
    if [ "$ready_count" -eq "$container_count" ] && [ "$total_pods" -eq "$DESIRED_PODS" ]; then
        pods_ready=true
        echo
        print_success "All $DESIRED_PODS pods are now ready!"
    else
        sleep 5
    fi
done

# Display detailed pod information
print_header "POD DETAILS"
echo "Here are the details of the running pods:"
check_pods

# Phase 2: Send second batch which should be distributed across pods
print_header "PHASE 2: TESTING LOAD DISTRIBUTION"
echo "Sending second batch of requests to demonstrate load distribution..."
send_batch 2 $((REQUESTS_PER_BATCH + 1)) $REQUESTS_PER_BATCH

# Analyze request distribution
analyze_request_distribution

print_header "DEMO SUMMARY"
echo "1. Initial state: 1 pod"
echo "2. Phase 1: Sent $REQUESTS_PER_BATCH requests, triggered scaling to $DESIRED_PODS pods"
echo "3. Phase 2: Sent another $REQUESTS_PER_BATCH requests, distributed across both pods"
echo "4. Final state: KServe will automatically scale down when load decreases"
echo
print_success "Demo completed successfully!"

print_header "NOTE"
echo "The service will automatically scale down when the load decreases."
echo "This typically takes a few minutes after the demo ends." 
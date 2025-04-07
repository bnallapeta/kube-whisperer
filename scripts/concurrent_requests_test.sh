#!/bin/bash

# Configuration
SERVICE_URL="http://kube-whisperer.default.74.224.102.71.nip.io"
AUDIO_FILE="/Users/bnr/work/github/whisper/samples/test.wav"

echo "Sending two concurrent requests to $SERVICE_URL..."
echo

# Send two requests simultaneously
curl -X POST "$SERVICE_URL/transcribe" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@$AUDIO_FILE" \
     -o response1.json &

curl -X POST "$SERVICE_URL/transcribe" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@$AUDIO_FILE" \
     -o response2.json &

curl -X POST "$SERVICE_URL/transcribe" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@$AUDIO_FILE" \
     -o response3.json &

curl -X POST "$SERVICE_URL/transcribe" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@$AUDIO_FILE" \
     -o response4.json &            

# Wait for both to complete
wait

echo
echo "Both responses received"
echo
echo "Response 1:"
cat response1.json
echo
echo "Response 2:"
cat response2.json 
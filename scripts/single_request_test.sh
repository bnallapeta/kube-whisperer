#!/bin/bash

# Configuration
SERVICE_URL="http://kube-whisperer.default.74.224.102.71.nip.io"
AUDIO_FILE="/Users/bnr/work/github/whisper/samples/test.wav"

echo "Sending a single request to $SERVICE_URL..."
echo

# Send request and save response to a file
curl -X POST "$SERVICE_URL/transcribe" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@$AUDIO_FILE" \
     -o response.json

echo
echo "Response received and saved to response.json"
echo
cat response.json
echo

# Send a second request after a 5-second delay
echo
echo "Waiting 5 seconds before sending a second request..."
sleep 5

echo "Sending second request..."
curl -X POST "$SERVICE_URL/transcribe" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@$AUDIO_FILE" \
     -o response2.json

echo
echo "Second response received and saved to response2.json"
echo
cat response2.json 
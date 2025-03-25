import requests
import argparse
import os
import time
import json
from typing import Optional, Dict, List

def test_health(base_url: str) -> bool:
    """Test the health endpoint."""
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        health_data = response.json()
        print("\nHealth check response:", json.dumps(health_data, indent=2))
        return health_data["status"] == "healthy"
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return False

def update_config(base_url: str, config: Dict) -> bool:
    """Update model configuration."""
    try:
        response = requests.post(f"{base_url}/config", json=config)
        response.raise_for_status()
        print("\nConfiguration update response:", json.dumps(response.json(), indent=2))
        return response.json()["status"] == "success"
    except Exception as e:
        print(f"Configuration update failed: {str(e)}")
        return False

def transcribe_audio(
    base_url: str,
    audio_file: str,
    options: Optional[Dict] = None
) -> dict:
    """
    Send audio file for transcription.
    
    Args:
        base_url (str): Base URL of the service
        audio_file (str): Path to audio file
        options (dict, optional): Transcription options
    
    Returns:
        dict: Transcription result
    """
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
    files = {
        'audio_file': open(audio_file, 'rb')
    }
    
    try:
        if options:
            response = requests.post(
                f"{base_url}/infer",
                files=files,
                json=options
            )
        else:
            response = requests.post(
                f"{base_url}/infer",
                files=files
            )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during transcription request: {str(e)}")
        if hasattr(e.response, 'json'):
            print("Error details:", e.response.json())
        raise
    finally:
        files['audio_file'].close()

def batch_transcribe(
    base_url: str,
    audio_files: List[str],
    options: Optional[Dict] = None
) -> dict:
    """
    Send multiple audio files for batch transcription.
    
    Args:
        base_url (str): Base URL of the service
        audio_files (List[str]): List of audio file paths
        options (dict, optional): Transcription options
    
    Returns:
        dict: Batch transcription results
    """
    request_data = {
        "files": audio_files
    }
    if options:
        request_data["options"] = options
    
    try:
        response = requests.post(
            f"{base_url}/batch",
            json=request_data
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during batch transcription request: {str(e)}")
        if hasattr(e.response, 'json'):
            print("Error details:", e.response.json())
        raise

def main():
    parser = argparse.ArgumentParser(description='Test Whisper Transcription Service')
    parser.add_argument('--url', default='http://localhost:8000', help='Service URL')
    parser.add_argument('--audio', required=True, help='Path to audio file')
    parser.add_argument('--batch', action='store_true', help='Test batch processing')
    parser.add_argument('--config', help='Path to config JSON file')
    parser.add_argument('--options', help='Path to transcription options JSON file')
    
    args = parser.parse_args()
    
    # Test health endpoint
    print("\nTesting health endpoint...")
    if not test_health(args.url):
        print("Health check failed. Exiting.")
        return
    
    # Update configuration if provided
    if args.config:
        print("\nUpdating configuration...")
        with open(args.config) as f:
            config = json.load(f)
        if not update_config(args.url, config):
            print("Configuration update failed. Exiting.")
            return
    
    # Load transcription options if provided
    options = None
    if args.options:
        with open(args.options) as f:
            options = json.load(f)
    
    # Test transcription
    if args.batch:
        print("\nTesting batch transcription...")
        # Use the provided audio file and create variations for batch testing
        audio_dir = os.path.dirname(args.audio)
        audio_files = [
            os.path.join(audio_dir, f)
            for f in os.listdir(audio_dir)
            if f.endswith(('.wav', '.mp3', '.ogg'))
        ]
        if not audio_files:
            print("No audio files found for batch processing")
            return
            
        try:
            start_time = time.time()
            result = batch_transcribe(args.url, audio_files, options)
            total_time = time.time() - start_time
            
            print("\nBatch Transcription Results:")
            print("-" * 50)
            for file_path, file_result in result["results"].items():
                print(f"\nFile: {os.path.basename(file_path)}")
                if "error" in file_result:
                    print(f"Error: {file_result['error']}")
                else:
                    print(f"Text: {file_result['text']}")
                    print(f"Language: {file_result['language']}")
            print(f"\nTotal Processing Time: {total_time:.2f}s")
            print("-" * 50)
            
        except Exception as e:
            print(f"Batch transcription test failed: {str(e)}")
    else:
        print("\nTesting single file transcription...")
        try:
            start_time = time.time()
            result = transcribe_audio(args.url, args.audio, options)
            total_time = time.time() - start_time
            
            print("\nTranscription Result:")
            print("-" * 50)
            print(f"Text: {result['text']}")
            print(f"Language: {result['language']}")
            if "segments" in result:
                print("\nSegments:")
                for segment in result["segments"]:
                    print(f"[{segment['start']:.1f}s -> {segment['end']:.1f}s] {segment['text']}")
            print(f"Processing Time: {result.get('processing_time', f'{total_time:.2f}s')}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Transcription test failed: {str(e)}")

if __name__ == "__main__":
    main() 
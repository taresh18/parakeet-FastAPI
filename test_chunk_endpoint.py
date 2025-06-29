#!/usr/bin/env python3
"""
Test script for comparing /v1/transcribe and /v1/transcribe-raw endpoints.
This script benchmarks both endpoints with real audio data and provides 
detailed latency comparison for real-time applications.
"""

import requests
import numpy as np
import sys
import time
import wave
from pathlib import Path
import statistics

def load_wav_as_raw_pcm(wav_file_path):
    """Load WAV file and extract raw PCM data with metadata."""
    try:
        with wave.open(str(wav_file_path), 'rb') as wav_file:
            # Get audio parameters
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            
            # Read raw PCM data
            raw_pcm_data = wav_file.readframes(n_frames)
            
            # Calculate duration
            duration = n_frames / sample_rate
            
            print(f"📊 Audio file info:")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Channels: {channels}")
            print(f"  Sample width: {sample_width} bytes ({sample_width * 8}-bit)")
            print(f"  Frames: {n_frames}")
            print(f"  Duration: {duration:.3f} seconds")
            print(f"  Raw PCM size: {len(raw_pcm_data)} bytes")
            
            return raw_pcm_data, sample_rate, channels, sample_width, duration
            
    except Exception as e:
        print(f"✗ Error loading WAV file: {e}")
        return None, None, None, None, None

def test_transcribe_chunk_endpoint(server_url, raw_pcm_data, sample_rate, duration, num_tests=5):
    """Test the /v1/transcribe endpoint with detailed latency analysis."""
    
    print(f"🎯 Testing /v1/transcribe endpoint")
    print("=" * 60)
    print(f"🔥 Starting warmup + measurement tests...")
    print("-" * 60)
    
    results = []
    transcriptions = []
    
    # Use session for keep-alive connections
    session = requests.Session()
    
    for i in range(num_tests + 1):  # +1 for warmup
        start_time = time.time()
        
        try:
            # Prepare form data with sample rate
            files = {'audio_data': ('audio.raw', raw_pcm_data, 'application/octet-stream')}
            data = {'sample_rate': sample_rate}
            
            response = session.post(
                f"{server_url}/v1/transcribe",
                files=files,
                data=data,
                timeout=30
            )
            
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                processing_time = result.get('processing_time', 0)
                network_time = total_time - processing_time
                transcription = result.get('text', '')
                
                if i == 0:  # Warmup
                    print(f"WARMUP: ✓ {total_time*1000:.1f}ms total ({processing_time*1000:.1f}ms processing + {network_time*1000:.1f}ms network)")
                    print(f"    📝 Text: \"{transcription[:200]}{'...' if len(transcription) > 200 else ''}\"")
                    print(f"    🔥 Warmup complete - model loaded and ready")
                else:
                    print(f"Test {i}/{num_tests}: ✓ {total_time*1000:.1f}ms total ({processing_time*1000:.1f}ms processing + {network_time*1000:.1f}ms network)")
                    print(f"    📝 Text: \"{transcription[:200]}{'...' if len(transcription) > 200 else ''}\"")
                    
                    results.append({
                        'total_time': total_time,
                        'processing_time': processing_time,
                        'network_time': network_time
                    })
                    transcriptions.append(transcription)
            else:
                print(f"Test {i}/{num_tests}: ✗ Failed with status {response.status_code}")
                print(f"    Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"Test {i}/{num_tests}: ✗ Exception: {e}")
            return None
    
    # Close session
    session.close()
    
    return results, transcriptions

def test_transcribe_raw_endpoint(server_url, raw_pcm_data, sample_rate, duration, num_tests=5):
    """Test the /v1/transcribe-raw endpoint with detailed latency analysis."""
    
    print(f"⚡ Testing /v1/transcribe-raw endpoint")
    print("=" * 60)
    print(f"🔥 Starting warmup + measurement tests...")
    print("-" * 60)
    
    results = []
    transcriptions = []
    
    # Use session for keep-alive connections
    session = requests.Session()
    
    for i in range(num_tests + 1):  # +1 for warmup
        start_time = time.time()
        
        try:
            # Send raw audio data directly in request body
            headers = {'Content-Type': 'application/octet-stream'}
            params = {'sample_rate': sample_rate}
            
            response = session.post(
                f"{server_url}/v1/transcribe-raw",
                data=raw_pcm_data,
                headers=headers,
                params=params,
                timeout=30
            )
            
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                processing_time = result.get('processing_time', 0)
                network_time = total_time - processing_time
                transcription = result.get('text', '')
                
                if i == 0:  # Warmup
                    print(f"WARMUP: ✓ {total_time*1000:.1f}ms total ({processing_time*1000:.1f}ms processing + {network_time*1000:.1f}ms network)")
                    print(f"    📝 Text: \"{transcription[:200]}{'...' if len(transcription) > 200 else ''}\"")
                    print(f"    🔥 Warmup complete - model loaded and ready")
                else:
                    print(f"Test {i}/{num_tests}: ✓ {total_time*1000:.1f}ms total ({processing_time*1000:.1f}ms processing + {network_time*1000:.1f}ms network)")
                    print(f"    📝 Text: \"{transcription[:200]}{'...' if len(transcription) > 200 else ''}\"")
                    
                    results.append({
                        'total_time': total_time,
                        'processing_time': processing_time,
                        'network_time': network_time
                    })
                    transcriptions.append(transcription)
            else:
                print(f"Test {i}/{num_tests}: ✗ Failed with status {response.status_code}")
                print(f"    Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"Test {i}/{num_tests}: ✗ Exception: {e}")
            return None
    
    # Close session
    session.close()
    
    return results, transcriptions

def test_transcribe_raw_canary_endpoint(server_url, raw_pcm_data, sample_rate, duration, num_tests=5):
    """Test the /v1/transcribe-raw/canary endpoint with detailed latency analysis."""
    
    print(f"🐦 Testing /v1/transcribe-raw/canary endpoint")
    print("=" * 60)
    print(f"🔥 Starting warmup + measurement tests...")
    print("-" * 60)
    
    results = []
    transcriptions = []
    
    # Use session for keep-alive connections
    session = requests.Session()
    
    for i in range(num_tests + 1):  # +1 for warmup
        start_time = time.time()
        
        try:
            # Send raw audio data directly in request body
            headers = {'Content-Type': 'application/octet-stream'}
            params = {'sample_rate': sample_rate}
            
            response = session.post(
                f"{server_url}/v1/transcribe-raw/canary",
                data=raw_pcm_data,
                headers=headers,
                params=params,
                timeout=30
            )
            
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                processing_time = result.get('processing_time', 0)
                network_time = total_time - processing_time
                transcription = result.get('text', '')
                
                if i == 0:  # Warmup
                    print(f"WARMUP: ✓ {total_time*1000:.1f}ms total ({processing_time*1000:.1f}ms processing + {network_time*1000:.1f}ms network)")
                    print(f"    📝 Text: \"{transcription[:200]}{'...' if len(transcription) > 200 else ''}\"")
                    print(f"    🔥 Warmup complete - model loaded and ready")
                else:
                    print(f"Test {i}/{num_tests}: ✓ {total_time*1000:.1f}ms total ({processing_time*1000:.1f}ms processing + {network_time*1000:.1f}ms network)")
                    print(f"    📝 Text: \"{transcription[:200]}{'...' if len(transcription) > 200 else ''}\"")
                    
                    results.append({
                        'total_time': total_time,
                        'processing_time': processing_time,
                        'network_time': network_time
                    })
                    transcriptions.append(transcription)
            else:
                print(f"Test {i}/{num_tests}: ✗ Failed with status {response.status_code}")
                print(f"    Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"Test {i}/{num_tests}: ✗ Exception: {e}")
            return None
    
    # Close session
    session.close()
    
    return results, transcriptions

def run_comparison_tests(server_url, audio_file, num_tests):
    """Run all comparison tests and return True if all succeed."""
    
    # Check if server is healthy
    try:
        health_start = time.time()
        health_response = requests.get(f"{server_url}/healthz", timeout=10)
        health_latency = time.time() - health_start
        
        if health_response.status_code == 200:
            print(f"✓ Server health check passed ({health_latency*1000:.1f}ms)")
        else:
            print(f"✗ Server health check failed: {health_response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Server health check failed: {e}")
        return False
    
    # Prepare test audio data
    audio_file_path = Path(audio_file)
    if not audio_file_path.exists():
        print(f"✗ Audio file not found: {audio_file}")
        return False
    
    # Load audio file for endpoints
    raw_pcm_data, sample_rate, channels, sample_width, audio_duration = load_wav_as_raw_pcm(audio_file_path)
    if raw_pcm_data is None:
        print("✗ Failed to load audio file")
        return False
    
    # Validate audio format
    if channels != 1:
        print(f"⚠️  Warning: Audio has {channels} channels, endpoints expect mono")
    if sample_width != 2:
        print(f"⚠️  Warning: Audio is {sample_width*8}-bit, endpoints expect 16-bit")
    
    print(f"\n📊 Test configuration:")
    print(f"  Audio data size: {len(raw_pcm_data):,} bytes")
    print(f"  Audio duration: {audio_duration:.3f} seconds")
    print(f"  Tests per endpoint: {num_tests} (+ 1 warmup)")
    
    try:
        # Test chunk endpoint
        chunk_results, chunk_transcriptions = test_transcribe_chunk_endpoint(server_url, raw_pcm_data, sample_rate, audio_duration, num_tests)
        
        # Small delay between endpoint tests
        print("\n" + "⏳" * 20)
        print("Pausing 2 seconds between endpoint tests...")
        time.sleep(2)
        
        # Test raw endpoint
        raw_results, raw_transcriptions = test_transcribe_raw_endpoint(server_url, raw_pcm_data, sample_rate, audio_duration, num_tests)
        
        # Small delay between endpoint tests
        print("\n" + "⏳" * 20)
        print("Pausing 2 seconds between endpoint tests...")
        time.sleep(2)
        
        # Test raw canary endpoint
        canary_results, canary_transcriptions = test_transcribe_raw_canary_endpoint(server_url, raw_pcm_data, sample_rate, audio_duration, num_tests)
        
        if not chunk_results or not raw_results or not canary_results:
            print("\n❌ One or more endpoint tests failed!")
            return False
            
        print("\n" + "📊" * 20)
        print("DETAILED PERFORMANCE COMPARISON")
        print("=" * 80)
        
        # Calculate statistics for chunk endpoint
        chunk_total_times = [r['total_time'] for r in chunk_results]
        chunk_processing_times = [r['processing_time'] for r in chunk_results]
        chunk_network_times = [r['network_time'] for r in chunk_results]
        
        # Calculate statistics for raw endpoint
        raw_total_times = [r['total_time'] for r in raw_results]
        raw_processing_times = [r['processing_time'] for r in raw_results]
        raw_network_times = [r['network_time'] for r in raw_results]
        
        # Calculate statistics for canary endpoint
        canary_total_times = [r['total_time'] for r in canary_results]
        canary_processing_times = [r['processing_time'] for r in canary_results]
        canary_network_times = [r['network_time'] for r in canary_results]
        
        def calculate_stats(times):
            if not times:
                return 0, 0, 0, 0, 0
            return (
                min(times) * 1000,
                max(times) * 1000, 
                statistics.mean(times) * 1000,
                statistics.median(times) * 1000,
                statistics.stdev(times) * 1000 if len(times) > 1 else 0
            )
        
        # Print detailed comparison
        print(f"🎯 /v1/transcribe Endpoint Performance:")
        chunk_min, chunk_max, chunk_avg, chunk_med, chunk_std = calculate_stats(chunk_total_times)
        print(f"   Total Latency:  {chunk_min:.1f}ms min, {chunk_max:.1f}ms max, {chunk_avg:.1f}ms avg, {chunk_med:.1f}ms median, ±{chunk_std:.1f}ms std")
        
        chunk_proc_min, chunk_proc_max, chunk_proc_avg, chunk_proc_med, chunk_proc_std = calculate_stats(chunk_processing_times)
        print(f"   Processing:     {chunk_proc_min:.1f}ms min, {chunk_proc_max:.1f}ms max, {chunk_proc_avg:.1f}ms avg, {chunk_proc_med:.1f}ms median, ±{chunk_proc_std:.1f}ms std")
        
        chunk_net_min, chunk_net_max, chunk_net_avg, chunk_net_med, chunk_net_std = calculate_stats(chunk_network_times)
        print(f"   Network:        {chunk_net_min:.1f}ms min, {chunk_net_max:.1f}ms max, {chunk_net_avg:.1f}ms avg, {chunk_net_med:.1f}ms median, ±{chunk_net_std:.1f}ms std")
        
        print(f"\n⚡ /v1/transcribe-raw Endpoint Performance:")
        raw_min, raw_max, raw_avg, raw_med, raw_std = calculate_stats(raw_total_times)
        print(f"   Total Latency:  {raw_min:.1f}ms min, {raw_max:.1f}ms max, {raw_avg:.1f}ms avg, {raw_med:.1f}ms median, ±{raw_std:.1f}ms std")
        
        raw_proc_min, raw_proc_max, raw_proc_avg, raw_proc_med, raw_proc_std = calculate_stats(raw_processing_times)
        print(f"   Processing:     {raw_proc_min:.1f}ms min, {raw_proc_max:.1f}ms max, {raw_proc_avg:.1f}ms avg, {raw_proc_med:.1f}ms median, ±{raw_proc_std:.1f}ms std")
        
        raw_net_min, raw_net_max, raw_net_avg, raw_net_med, raw_net_std = calculate_stats(raw_network_times)
        print(f"   Network:        {raw_net_min:.1f}ms min, {raw_net_max:.1f}ms max, {raw_net_avg:.1f}ms avg, {raw_net_med:.1f}ms median, ±{raw_net_std:.1f}ms std")
        
        print(f"\n🐦 /v1/transcribe-raw/canary Endpoint Performance:")
        canary_min, canary_max, canary_avg, canary_med, canary_std = calculate_stats(canary_total_times)
        print(f"   Total Latency:  {canary_min:.1f}ms min, {canary_max:.1f}ms max, {canary_avg:.1f}ms avg, {canary_med:.1f}ms median, ±{canary_std:.1f}ms std")
        
        canary_proc_min, canary_proc_max, canary_proc_avg, canary_proc_med, canary_proc_std = calculate_stats(canary_processing_times)
        print(f"   Processing:     {canary_proc_min:.1f}ms min, {canary_proc_max:.1f}ms max, {canary_proc_avg:.1f}ms avg, {canary_proc_med:.1f}ms median, ±{canary_proc_std:.1f}ms std")
        
        canary_net_min, canary_net_max, canary_net_avg, canary_net_med, canary_net_std = calculate_stats(canary_network_times)
        print(f"   Network:        {canary_net_min:.1f}ms min, {canary_net_max:.1f}ms max, {canary_net_avg:.1f}ms avg, {canary_net_med:.1f}ms median, ±{canary_net_std:.1f}ms std")
        
        # Performance comparison
        print(f"\n⚡ PERFORMANCE COMPARISON:")
        raw_improvement = ((chunk_avg - raw_avg) / chunk_avg) * 100 if chunk_avg > 0 else 0
        
        if raw_avg < chunk_avg:
            print(f"   🚀 /v1/transcribe-raw is {raw_improvement:.1f}% FASTER than /v1/transcribe ({chunk_avg - raw_avg:.1f}ms saved per request)")
        else:
            print(f"   📉 /v1/transcribe-raw is {-raw_improvement:.1f}% SLOWER than /v1/transcribe ({raw_avg - chunk_avg:.1f}ms additional per request)")
        
        canary_improvement_vs_chunk = ((chunk_avg - canary_avg) / chunk_avg) * 100 if chunk_avg > 0 else 0
        if canary_avg < chunk_avg:
            print(f"   🚀 /v1/transcribe-raw/canary is {canary_improvement_vs_chunk:.1f}% FASTER than /v1/transcribe ({chunk_avg - canary_avg:.1f}ms saved per request)")
        else:
            print(f"   📉 /v1/transcribe-raw/canary is {-canary_improvement_vs_chunk:.1f}% SLOWER than /v1/transcribe ({canary_avg - chunk_avg:.1f}ms additional per request)")
        
        canary_improvement_vs_raw = ((raw_avg - canary_avg) / raw_avg) * 100 if raw_avg > 0 else 0
        if canary_avg < raw_avg:
            print(f"   🚀 /v1/transcribe-raw/canary is {canary_improvement_vs_raw:.1f}% FASTER than /v1/transcribe-raw ({raw_avg - canary_avg:.1f}ms saved per request)")
        else:
            print(f"   📉 /v1/transcribe-raw/canary is {-canary_improvement_vs_raw:.1f}% SLOWER than /v1/transcribe-raw ({canary_avg - raw_avg:.1f}ms additional per request)")
        
        # Real-time factors
        chunk_rt_factor = (audio_duration / (chunk_proc_avg / 1000)) if chunk_proc_avg > 0 else 0
        raw_rt_factor = (audio_duration / (raw_proc_avg / 1000)) if raw_proc_avg > 0 else 0
        canary_rt_factor = (audio_duration / (canary_proc_avg / 1000)) if canary_proc_avg > 0 else 0
        
        print(f"\n🎵 REAL-TIME PERFORMANCE:")
        print(f"   Audio duration:     {audio_duration:.3f} seconds")
        print(f"   /v1/transcribe:     {chunk_rt_factor:.0f}x real-time")
        print(f"   /v1/transcribe-raw: {raw_rt_factor:.0f}x real-time")
        print(f"   /v1/transcribe-raw/canary: {canary_rt_factor:.0f}x real-time")
        
        # Transcription consistency check
        print(f"\n📝 TRANSCRIPTION CONSISTENCY:")
        chunk_unique = len(set(chunk_transcriptions))
        raw_unique = len(set(raw_transcriptions))
        canary_unique = len(set(canary_transcriptions))
        
        print(f"   /v1/transcribe:     {chunk_unique}/{len(chunk_transcriptions)} unique results")
        print(f"   /v1/transcribe-raw: {raw_unique}/{len(raw_transcriptions)} unique results")
        print(f"   /v1/transcribe-raw/canary: {canary_unique}/{len(canary_transcriptions)} unique results")
        
        if chunk_unique == 1 and raw_unique == 1 and canary_unique == 1:
            if chunk_transcriptions[0] == raw_transcriptions[0] and raw_transcriptions[0] == canary_transcriptions[0]:
                print(f"   ✅ All endpoints produce identical results")
            else:
                print(f"   ⚠️  Endpoints produce different results:")
                print(f"      /v1/transcribe:     \"{chunk_transcriptions[0][:100]}{'...' if len(chunk_transcriptions[0]) > 100 else ''}\"")
                print(f"      /v1/transcribe-raw: \"{raw_transcriptions[0][:100]}{'...' if len(raw_transcriptions[0]) > 100 else ''}\"")
                print(f"      /v1/transcribe-raw/canary: \"{canary_transcriptions[0][:100]}{'...' if len(canary_transcriptions[0]) > 100 else ''}\"")
        else:
            print(f"   ⚠️  Inconsistent results detected")
        
        return True
    
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False

def main():
    """Main function to run endpoint comparison tests."""
    
    # Default parameters
    server_url = "http://localhost:8989"
    audio_file = "output_24000khz.wav"
    num_tests = 5
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    if len(sys.argv) > 2:
        audio_file = sys.argv[2]
    if len(sys.argv) > 3:
        num_tests = int(sys.argv[3])
    
    print(f"🎤 Enhanced Endpoint Comparison Tool")
    print(f"Usage: {sys.argv[0]} [server_url] [audio_file] [num_tests]")
    print(f"Server URL: {server_url}")
    print(f"Audio file: {audio_file}")
    print(f"Number of tests: {num_tests}")
    print()
    
    success = run_comparison_tests(server_url, audio_file, num_tests)
    
    if success:
        print("\n✅ All tests completed successfully!")
        print("🚀 Performance analysis complete!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
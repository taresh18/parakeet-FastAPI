import requests
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

def test_transcribe_parakeet_endpoint(server_url, raw_pcm_data, sample_rate, num_tests=5):
    """Test the /v1/transcribe/parakeet endpoint with detailed latency analysis."""
    
    print(f"🎯 Testing /v1/transcribe/parakeet endpoint")
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
                f"{server_url}/v1/transcribe/parakeet",
                data=raw_pcm_data,
                headers=headers,
                params=params,
                timeout=30
            )
            
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                server_processing_time = result.get('processing_time', 0)
                network_time = total_time - server_processing_time
                transcription = result.get('text', '')
                
                if i == 0:  # Warmup
                    print(f"WARMUP: ✓ {total_time*1000:.1f}ms total ({server_processing_time*1000:.1f}ms server + {network_time*1000:.1f}ms network)")
                    print(f"    📝 Text: \"{transcription[:200]}{'...' if len(transcription) > 200 else ''}\"")
                    print(f"    🔥 Warmup complete - model loaded and ready")
                else:
                    results.append({
                        'total_time': total_time,
                        'server_processing_time': server_processing_time,
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

def test_transcribe_canary_endpoint(server_url, raw_pcm_data, sample_rate, num_tests=5):
    """Test the /v1/transcribe/canary endpoint with detailed latency analysis."""
    
    print(f"🐦 Testing /v1/transcribe/canary endpoint")
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
                f"{server_url}/v1/transcribe/canary",
                data=raw_pcm_data,
                headers=headers,
                params=params,
                timeout=30
            )
            
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                server_processing_time = result.get('processing_time', 0)
                network_time = total_time - server_processing_time
                transcription = result.get('text', '')
                
                if i == 0:  # Warmup
                    print(f"WARMUP: ✓ {total_time*1000:.1f}ms total ({server_processing_time*1000:.1f}ms server + {network_time*1000:.1f}ms network)")
                    print(f"    📝 Text: \"{transcription[:200]}{'...' if len(transcription) > 200 else ''}\"")
                    print(f"    🔥 Warmup complete - model loaded and ready")
                else:
                    results.append({
                        'total_time': total_time,
                        'server_processing_time': server_processing_time,
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
    
    print(f"\n📊 Test configuration:")
    print(f"  Audio data size: {len(raw_pcm_data):,} bytes")
    print(f"  Audio duration: {audio_duration:.3f} seconds")
    print(f"  Tests per endpoint: {num_tests} (+ 1 warmup)")
    
    try:
        # Test parakeet endpoint
        parakeet_results, parakeet_transcriptions = test_transcribe_parakeet_endpoint(server_url, raw_pcm_data, sample_rate, num_tests)
        
        # Small delay between endpoint tests
        print("\n" + "⏳" * 20)
        print("Pausing 2 seconds between endpoint tests...")
        time.sleep(2)
        
        # Test canary endpoint
        canary_results, canary_transcriptions = test_transcribe_canary_endpoint(server_url, raw_pcm_data, sample_rate, num_tests)
        
        if not parakeet_results or not canary_results:
            print("\n❌ One or more endpoint tests failed!")
            return False
            
        print("\n" + "📊" * 20)
        print("DETAILED PERFORMANCE COMPARISON")
        print("=" * 80)
        
        # Calculate statistics for parakeet endpoint
        parakeet_total_times = [r['total_time'] for r in parakeet_results]
        parakeet_processing_times = [r['server_processing_time'] for r in parakeet_results]
        parakeet_network_times = [r['network_time'] for r in parakeet_results]
        
        # Calculate statistics for canary endpoint
        canary_total_times = [r['total_time'] for r in canary_results]
        canary_processing_times = [r['server_processing_time'] for r in canary_results]
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
        print(f"🎯 /v1/transcribe/parakeet Endpoint Performance:")
        parakeet_min, parakeet_max, parakeet_avg, parakeet_med, parakeet_std = calculate_stats(parakeet_total_times)
        print(f"   Total Latency:  {parakeet_min:.1f}ms min, {parakeet_max:.1f}ms max, {parakeet_avg:.1f}ms avg, {parakeet_med:.1f}ms median, ±{parakeet_std:.1f}ms std")
        
        parakeet_proc_min, parakeet_proc_max, parakeet_proc_avg, parakeet_proc_med, parakeet_proc_std = calculate_stats(parakeet_processing_times)
        print(f"   Server Processing: {parakeet_proc_min:.1f}ms min, {parakeet_proc_max:.1f}ms max, {parakeet_proc_avg:.1f}ms avg, {parakeet_proc_med:.1f}ms median, ±{parakeet_proc_std:.1f}ms std")
        
        parakeet_net_min, parakeet_net_max, parakeet_net_avg, parakeet_net_med, parakeet_net_std = calculate_stats(parakeet_network_times)
        print(f"   Network Overhead: {parakeet_net_min:.1f}ms min, {parakeet_net_max:.1f}ms max, {parakeet_net_avg:.1f}ms avg, {parakeet_net_med:.1f}ms median, ±{parakeet_net_std:.1f}ms std")
        
        print(f"\n🐦 /v1/transcribe/canary Endpoint Performance:")
        canary_min, canary_max, canary_avg, canary_med, canary_std = calculate_stats(canary_total_times)
        print(f"   Total Latency:  {canary_min:.1f}ms min, {canary_max:.1f}ms max, {canary_avg:.1f}ms avg, {canary_med:.1f}ms median, ±{canary_std:.1f}ms std")
        
        canary_proc_min, canary_proc_max, canary_proc_avg, canary_proc_med, canary_proc_std = calculate_stats(canary_processing_times)
        print(f"   Server Processing: {canary_proc_min:.1f}ms min, {canary_proc_max:.1f}ms max, {canary_proc_avg:.1f}ms avg, {canary_proc_med:.1f}ms median, ±{canary_proc_std:.1f}ms std")
        
        canary_net_min, canary_net_max, canary_net_avg, canary_net_med, canary_net_std = calculate_stats(canary_network_times)
        print(f"   Network Overhead: {canary_net_min:.1f}ms min, {canary_net_max:.1f}ms max, {canary_net_avg:.1f}ms avg, {canary_net_med:.1f}ms median, ±{canary_net_std:.1f}ms std")
        
        # Performance comparison
        print(f"\n⚡ PERFORMANCE COMPARISON:")
        canary_improvement_vs_parakeet = ((parakeet_avg - canary_avg) / parakeet_avg) * 100 if parakeet_avg > 0 else 0
        
        if canary_avg < parakeet_avg:
            print(f"   🚀 /v1/transcribe/canary is {canary_improvement_vs_parakeet:.1f}% FASTER than /v1/transcribe/parakeet ({parakeet_avg - canary_avg:.1f}ms saved per request)")
        else:
            print(f"   📉 /v1/transcribe/canary is {-canary_improvement_vs_parakeet:.1f}% SLOWER than /v1/transcribe/parakeet ({canary_avg - parakeet_avg:.1f}ms additional per request)")
        
        # Network overhead comparison
        network_diff = canary_net_avg - parakeet_net_avg
        if network_diff > 0:
            print(f"   🌐 /v1/transcribe/canary has {network_diff:.1f}ms MORE network overhead than /v1/transcribe/parakeet")
        else:
            print(f"   🌐 /v1/transcribe/canary has {-network_diff:.1f}ms LESS network overhead than /v1/transcribe/parakeet")
        
        # Server processing comparison
        processing_diff = canary_proc_avg - parakeet_proc_avg
        if processing_diff > 0:
            print(f"   ⚙️  /v1/transcribe/canary takes {processing_diff:.1f}ms LONGER to process than /v1/transcribe/parakeet")
        else:
            print(f"   ⚙️  /v1/transcribe/canary takes {-processing_diff:.1f}ms LESS to process than /v1/transcribe/parakeet")
        
        # Real-time factors
        parakeet_rt_factor = (audio_duration / (parakeet_proc_avg / 1000)) if parakeet_proc_avg > 0 else 0
        canary_rt_factor = (audio_duration / (canary_proc_avg / 1000)) if canary_proc_avg > 0 else 0
        
        print(f"\n🎵 REAL-TIME PERFORMANCE:")
        print(f"   Audio duration:     {audio_duration:.3f} seconds")
        print(f"   /v1/transcribe/parakeet: {parakeet_rt_factor:.0f}x real-time")
        print(f"   /v1/transcribe/canary: {canary_rt_factor:.0f}x real-time")
        
        # Transcription consistency check
        print(f"\n📝 TRANSCRIPTION CONSISTENCY:")
        parakeet_unique = len(set(parakeet_transcriptions))
        canary_unique = len(set(canary_transcriptions))
        
        print(f"   /v1/transcribe/parakeet: {parakeet_unique}/{len(parakeet_transcriptions)} unique results")
        print(f"   /v1/transcribe/canary: {canary_unique}/{len(canary_transcriptions)} unique results")
        
        if parakeet_unique == 1 and canary_unique == 1:
            if parakeet_transcriptions[0] == canary_transcriptions[0]:
                print(f"   ✅ Both endpoints produce identical results")
            else:
                print(f"   ⚠️  Endpoints produce different results:")
                print(f"      /v1/transcribe/parakeet: \"{parakeet_transcriptions[0][:100]}{'...' if len(parakeet_transcriptions[0]) > 100 else ''}\"")
                print(f"      /v1/transcribe/canary: \"{canary_transcriptions[0][:100]}{'...' if len(canary_transcriptions[0]) > 100 else ''}\"")
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
    server_url = "http://0.0.0.0:8989"
    audio_file = "test_audio.wav"
    num_tests = 5
    
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
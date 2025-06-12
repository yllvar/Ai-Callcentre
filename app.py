import os
import wave
import pyaudio
import numpy as np
import time  # Make sure this is imported
from scipy.io import wavfile
from faster_whisper import WhisperModel
import voice_service as vs
from rag.AIVoiceAssistant import AIVoiceAssistant

DEFAULT_MODEL_SIZE = "medium"
DEFAULT_CHUNK_LENGTH = 5  # Reduced from 10 to make more responsive

ai_assistant = AIVoiceAssistant()

def is_silence(data, max_amplitude_threshold=3000):
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold

def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        try:
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        except OSError:
            return True  # Treat overflow as silence to restart stream

    temp_file_path = 'temp_audio_chunk.wav'
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True
        return False
    except Exception:
        return False

def transcribe_audio(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=5)  # Reduced beam_size for speed
    return ' '.join(segment.text for segment in segments)

def main():
    model_size = DEFAULT_MODEL_SIZE + ".en"
    model = WhisperModel(model_size, device="cpu", compute_type="int8", num_workers=2)  # Reduced workers
    
    audio = pyaudio.PyAudio()
    stream = None
    
    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=2048,
            input_device_index=audio.get_default_input_device_info()['index'],
            start=True
        )
        
        conversation_active = False
        
        while True:
            try:
                chunk_file = "temp_audio_chunk.wav"
                
                # Record shorter chunks for more responsive interaction
                silence = record_audio_chunk(audio, stream, chunk_length=3)  # Reduced chunk length
                
                if not silence:
                    transcription = transcribe_audio(model, chunk_file)
                    if os.path.exists(chunk_file):
                        os.remove(chunk_file)
                    
                    if transcription.strip():
                        print(f"Customer: {transcription}")
                        
                        # Get assistant response
                        output = ai_assistant.interact_with_llm(transcription)
                        if output:
                            output = output.strip()
                            print(f"AI Assistant: {output}")
                            
                            # Play response without waiting for full completion
                            vs.play_text_to_speech(output, wait_for_completion=False)
                            
                            # Short pause to prevent talking over user
                            time.sleep(0.5)
                            
                        conversation_active = True
                    else:
                        if conversation_active:
                            print("AI Assistant: Is there anything else I can help you with?")
                            vs.play_text_to_speech("Is there anything else I can help you with?")
                            conversation_active = False
                else:
                    if conversation_active:
                        print("AI Assistant: I didn't hear anything. Can you please repeat?")
                        vs.play_text_to_speech("I didn't hear anything. Can you please repeat?")
                        conversation_active = False

            except OSError as e:
                if e.errno == -9981:  # Input overflow
                    if stream:
                        stream.stop_stream()
                        stream.start_stream()
                continue
                
            except Exception as e:
                print(f"Error: {str(e)}")
                continue

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            if stream:
                stream.stop_stream()
                stream.close()
            audio.terminate()
        except:
            pass

if __name__ == "__main__":
    main()
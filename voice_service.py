import os
import time
import pygame
from gtts import gTTS
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def play_text_to_speech(text, language='en', slow=False, temp_audio_file="temp_audio.mp3", wait_for_completion=False):
    def _play():
        try:
            if not text.strip():
                return
                
            tts = gTTS(text=text, lang=language, slow=slow)
            tts.save(temp_audio_file)
            
            pygame.mixer.init()
            pygame.mixer.music.load(temp_audio_file)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
        except Exception as e:
            print(f"TTS Error: {str(e)}")
        finally:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                if os.path.exists(temp_audio_file):
                    os.remove(temp_audio_file)
            except:
                pass
    
    # Run in thread to avoid blocking
    thread = threading.Thread(target=_play)
    thread.start()
    
    if wait_for_completion:
        thread.join()
        
# Example usage
if __name__ == "__main__":
    play_text_to_speech("Hello, this is a test of the text-to-speech service.")

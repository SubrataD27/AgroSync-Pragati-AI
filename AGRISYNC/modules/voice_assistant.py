import os
import tempfile
import numpy as np
import streamlit as st
from gtts import gTTS
import speech_recognition as sr
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from io import BytesIO
import base64

# Language mapping
LANGUAGE_CODES = {
    "English": "en",
    "हिंदी (Hindi)": "hi",
    "தமிழ் (Tamil)": "ta",
    "తెలుగు (Telugu)": "te",
    "ಕನ್ನಡ (Kannada)": "kn",
    "മലയാളം (Malayalam)": "ml",
    "ਪੰਜਾਬੀ (Punjabi)": "pa"
}

class VoiceAssistant:
    """Voice assistant for speech recognition and synthesis"""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_whisper_model(self):
        """Load the Whisper model for speech recognition"""
        if self.processor is None or self.model is None:
            with st.spinner("Loading voice recognition model..."):
                # Using small model for efficiency
                model_name = "openai/whisper-small"
                self.processor = WhisperProcessor.from_pretrained(model_name)
                self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
                
                # Set forced decoder ids for language
                self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language="english", task="transcribe"
                )
                
    def set_language(self, language):
        """Set the language for the voice assistant"""
        if language in LANGUAGE_CODES:
            lang_code = LANGUAGE_CODES[language]
            # Update whisper model language setting
            if self.model is not None:
                self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language.split(" ")[0].lower(), task="transcribe"
                )
            return lang_code
        return "en"  # Default to English
    
    def recognize_speech(self, audio_file, language="English"):
        """Recognize speech from audio file"""
        self.load_whisper_model()
        
        try:
            # Load audio data
            audio_data, _ = librosa.load(audio_file, sr=16000)
            
            # Process audio with whisper
            input_features = self.processor(
                audio_data, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate token ids
            predicted_ids = self.model.generate(input_features)
            
            # Decode token ids to text
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription
        except Exception as e:
            st.error(f"Error recognizing speech: {e}")
            return None
    
    def recognize_from_microphone(self, language="English"):
        """Record and recognize speech from microphone"""
        recognizer = sr.Recognizer()
        
        with st.spinner("Listening..."):
            try:
                with sr.Microphone() as source:
                    st.write("Say something...")
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source, timeout=5)
                    
                    # Save audio to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        f.write(audio.get_wav_data())
                        temp_filename = f.name
                    
                    # Process using Whisper
                    transcription = self.recognize_speech(temp_filename, language)
                    
                    # Clean up
                    os.unlink(temp_filename)
                    
                    return transcription
                    
            except sr.RequestError as e:
                st.error(f"Could not request results: {e}")
                return None
            except sr.UnknownValueError:
                st.error("Could not understand audio")
                return None
            except Exception as e:
                st.error(f"Error: {e}")
                return None

    def text_to_speech(self, text, language="English"):
        """Convert text to speech"""
        if not text:
            return None
        
        try:
            lang_code = self.set_language(language)
            
            # Create gTTS object
            tts = gTTS(text=text, lang=lang_code, slow=False)
            
            # Save to BytesIO object
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            # Create HTML with audio player
            audio_bytes = fp.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()
            audio_tag = f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}" controls></audio>'
            
            return audio_tag
        except Exception as e:
            st.error(f"Text-to-speech error: {e}")
            return None

# Singleton instance
voice_assistant = VoiceAssistant()

def get_voice_assistant():
    """Get the voice assistant instance"""
    return voice_assistant
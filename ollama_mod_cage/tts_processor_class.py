import threading
import os
import re
import queue
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import shutil
from TTS.api import TTS
import keyboard

class tts_processor_class:
    def __init__(self):
        self.current_dir = os.getcwd()
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))
        self.speech_dir = os.path.join(self.parent_dir, "AgentFiles", "pipeline", "speech_library")
        self.recognize_speech_dir = os.path.join(self.parent_dir, "AgentFiles", "pipeline", "speech_library", "recognize_speech")
        self.generate_speech_dir = os.path.join(self.parent_dir, "AgentFiles", "pipeline", "speech_library", "generate_speech")
        self.tts_voice_ref_wav_pack_path = os.path.join(self.parent_dir, "AgentFiles", "pipeline", "active_group", "Public_Voice_Reference_Pack")
        self.conversation_library = os.path.join(self.parent_dir, "AgentFiles", "pipeline", "conversation_library")
        self.interrupt_flag = False  # Flag to interrupt audio generation

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            print("CUDA-compatible GPU is not available. Using CPU instead. If you believe this should not be the case, reinstall torch-audio with the correct version.")

        # Replace with your actual TTS model initialization
        self.tts = TTS(model_path="E:/AI/ollama_agent_roll_cage/AgentFiles/Ignored_TTS/PeterDrury_Podcast/",
                       config_path="E:/AI/ollama_agent_roll_cage/AgentFiles/Ignored_TTS/PeterDrury_Podcast/config.json",
                       progress_bar=False, gpu=False).to(self.device)
        self.audio_queue = queue.Queue()

            # Register the hotkey (Ctrl + R) to set interrupt_flag
        keyboard.add_hotkey('ctrl+r', self.handle_interrupt)

    def handle_interrupt(self):
        """ Method to handle interrupt keybind (Ctrl + R). """
        print("Interrupting audio generation...")
        self.interrupt_flag = True

    def get_audio(self):
        """ A method for collecting audio from the microphone. """
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = r.listen(source)
        return audio

    def recognize_speech(self, audio):
        """ A method for calling the speech recognizer. """
        speech_str = sr.Recognizer().recognize_google(audio, language='en-US')
        print(f">>{speech_str}<<")
        return speech_str

    def process_tts_responses(self, response, voice_name):
        """ A method for managing the response preprocessing methods. """
        torch.cuda.empty_cache()  # Clear VRAM cache
        tts_response_sentences = self.split_into_sentences(response)
        self.clear_directory(self.recognize_speech_dir)
        self.clear_directory(self.generate_speech_dir)
        self.generate_play_audio_loop(tts_response_sentences, voice_name)

    def play_audio_from_file(self, filename):
        """ A method for audio playback from file. """
        if not os.path.isfile(filename):
            print(f"File {filename} does not exist.")
            return

        try:
            audio_data, sample_rate = sf.read(filename)
            sd.play(audio_data, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Failed to play audio from file {filename}. Reason: {e}")

    def generate_audio(self, sentence, voice_name_path, ticker):
        """ A method to generate audio for the chatbot. """
        print("Starting speech generation...")
        try:
            tts_audio = self.tts.tts(text=sentence, speaker_wav=voice_name_path, language="en", speed=3)
            tts_audio = np.array(tts_audio, dtype=np.float32)
            filename = os.path.join(self.generate_speech_dir, f"audio_{ticker}.wav")
            sf.write(filename, tts_audio, 22050)
        except Exception as e:
            print(f"Failed to generate audio for sentence '{sentence}'. Reason: {e}")

    def generate_play_audio_loop(self, tts_response_sentences, voice_name):
        """ A method for generating and playing audio in a loop. """
        self.interrupt_flag = False
        ticker = 0
        voice_name_path = os.path.join(self.tts_voice_ref_wav_pack_path, f"{voice_name}\\clone_speech.wav")
        audio_thread = threading.Thread(target=self.generate_audio,
                                        args=(tts_response_sentences[0], voice_name_path, ticker))
        audio_thread.start()

        try:
            while not self.interrupt_flag:
                audio_thread.join()
                if self.interrupt_flag:
                    break
                filename = os.path.join(self.generate_speech_dir, f"audio_{ticker}.wav")
                play_thread = threading.Thread(target=self.play_audio_from_file, args=(filename,))
                play_thread.start()
                ticker += 1
                audio_thread = threading.Thread(target=self.generate_audio,
                                                args=(tts_response_sentences[ticker], voice_name_path, ticker))
                audio_thread.start()
                play_thread.join()

            audio_thread.join()
            filename = os.path.join(self.generate_speech_dir, f"audio_{ticker}.wav")
            self.play_audio_from_file(filename)

        except Exception as e:
            print(f"Exception occurred in generate_play_audio_loop: {e}")

        finally:
            print("Audio generation and playback loop ended.")

    def clear_directory(self, directory):
        """ A method to clear files in a directory. """
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    def split_into_sentences(self, text):
        """ A method to split text into sentences. """
        text = " " + text + " "
        text = text.replace("\n", " ")
        text = re.sub(r"(Mr|Mrs|Ms|Dr|i\.e)\.", r"\1<prd>", text)
        text = re.sub(r"\.\.\.", r"<prd><prd><prd>", text)
        sentences = re.split(r"(?<=\d\.)\s+|(?<=[.!?:])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        combined_sentences = []
        i = 0
        while i < len(sentences):
            if re.match(r"^\d+\.", sentences[i]):
                combined_sentences.append(f"{sentences[i]} {sentences[i + 1]}")
                i += 2
            else:
                combined_sentences.append(sentences[i])
                i += 1
        return combined_sentences

    def file_name_voice_filter(self, input):
        """ A method to filter filenames for voice references. """
        return re.sub(' ', '_', input).lower()

    def file_name_conversation_history_filter(self, input):
        """ A method to filter filenames for conversation history. """
        return re.sub(' ', '_', input).lower()

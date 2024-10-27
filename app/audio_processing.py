import io
import librosa
import soundfile as sf
import numpy as np
import base64
from typing import Union
from numpy.typing import NDArray
from pydub import AudioSegment

def process_audio(contents: bytes, operation: str = 'original') -> str:
    # Convert the audio to WAV format using pydub
    audio = AudioSegment.from_file(io.BytesIO(contents))
    wav_io = io.BytesIO()
    audio.export(wav_io, format='wav')
    wav_io.seek(0)

    # Load the audio data using librosa
    y, sr = librosa.load(wav_io, sr=None)

    # Apply the selected operation
    if operation == 'noise_reduction':
        y = librosa.effects.preemphasis(y)
    elif operation == 'pitch_shift':
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    elif operation == 'time_stretch':
        y = librosa.effects.time_stretch(y, rate=1.2)
    elif operation == 'reverb':
        y = np.concatenate([y, librosa.effects.preemphasis(y)])
    
    # Convert the processed audio to WAV format
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format='wav')
    buffer.seek(0)
    
    # Encode the WAV data to base64
    audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return audio_base64

import io
import librosa
import numpy as np
import base64
from typing import Tuple, Union
from numpy.typing import NDArray

def process_audio(contents: bytes, operation: str = 'original') -> str:
    y, sr = librosa.load(io.BytesIO(contents))
    
    if operation == 'noise_reduction':
        y = librosa.effects.preemphasis(y)
    elif operation == 'pitch_shift':
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    elif operation == 'time_stretch':
        y = librosa.effects.time_stretch(y, rate=1.2)
    elif operation == 'reverb':
        y = np.concatenate([y, librosa.effects.preemphasis(y)])
    
    audio_bytes = librosa.util.buf_to_float(y)
    audio_base64 = base64.b64encode(audio_bytes.tobytes()).decode('utf-8')
    return audio_base64

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torchaudio
from torchaudio.transforms import Resample, Spectrogram, TimeStretch, TimeMasking, FrequencyMasking, MelScale
import numpy as np
import librosa
import librosa.display
import os
from IPython.display import display, Audio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np  
import sys


class LogMagSpectrogramPipeline(nn.Module):
    def __init__(self, 
                 sample_rate=44100, 
                 n_fft=2048, 
                 hop_size_ms=10, 
                 f_min=30.0, 
                 f_max=17000.0, 
                 n_bands=81):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        
        # Calculate hop length in samples (44.1kHz * 10ms = 441 samples)
        self.hop_length = int(sample_rate * (hop_size_ms / 1000.0))
        
        # Create a Hann window and register it as a buffer (moves to GPU automatically)
        self.register_buffer('window', torch.hann_window(n_fft))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Pre-compute the logarithmic filterbank matrix
        filterbank = self._create_log_filterbank(
            sample_rate=sample_rate, 
            n_fft=n_fft, 
            n_bands=n_bands, 
            f_min=f_min, 
            f_max=f_max
        )
        self.register_buffer('filterbank', filterbank)

    def _create_log_filterbank(self, sample_rate, n_fft, n_bands, f_min, f_max):
        """
        Creates a triangular filterbank matrix with logarithmic frequency spacing.
        """
        # Generate logarithmically spaced center frequencies
        # n_bands + 2 points to create overlapping triangular filters
        log_freqs = np.logspace(np.log10(f_min), np.log10(f_max), num=n_bands + 2)
        
        # Linear frequencies of the STFT bins
        fft_freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        
        # Initialize the filterbank matrix
        fb = np.zeros((n_bands, n_fft // 2 + 1), dtype=np.float32)
        
        for i in range(n_bands):
            f_left = log_freqs[i]
            f_center = log_freqs[i + 1]
            f_right = log_freqs[i + 2]
            
            # Create a triangular filter for the current band
            # It rises linearly from f_left to f_center, and falls to f_right
            fb[i] = np.maximum(0, np.minimum(
                (fft_freqs - f_left) / (f_center - f_left),
                (f_right - fft_freqs) / (f_right - f_center)
            ))
            
        return torch.tensor(fb, dtype=torch.float32)

    def forward(self, waveform):
        """
        Args:
            waveform: Tensor of shape (batch_size, num_samples) or (num_samples,)
        Returns:
            log_mag_spec: Tensor of shape (batch_size, n_bands, num_frames)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform.to(self.device)
        # 1. Compute the STFT
        # center=True pads the signal so frames are centered at hop intervals
        stft_out = torch.stft(
            waveform, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.n_fft, 
            window=self.window, 
            center=True, 
            pad_mode='reflect', 
            return_complex=True
        )
        
        # 2. Convert to magnitude spectrogram
        mag_spec = torch.abs(stft_out)
        
        # 3. Apply the logarithmic filterbank
        # Matrix multiplication: (n_bands, num_fft_bins) x (batch_size, num_fft_bins, num_frames)
        # Resulting shape: (batch_size, n_bands, num_frames)
        filtered_spec = torch.matmul(self.filterbank, mag_spec)
        
        # 4. Convert to Log Magnitude (adding a small epsilon to prevent log(0))
        log_mag_spec = torch.log10(filtered_spec + 1e-6)
        
        return log_mag_spec
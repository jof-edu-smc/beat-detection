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

def plot_waveform(waveform, sr, 
                  ground_truth=None, 
                  beats=None,
                  downbeats=None, 
                  sample_rate=44100, 
                  hop_length=441, 
                  title="Waveform", 
                  ax=None
                ):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1, figsize=(20, 3 * num_channels), sharex=True)
    ax.plot(time_axis, waveform[0], linewidth=1, color="black")
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)
    # Ground-truth lines
    converted_beats = []
    if ground_truth is not None:
        # print(f"Ground Truth: {ground_truth}")
        for i, beat in enumerate(ground_truth):
            if beat > 0.5: 
                beat_time = (i * hop_length) / sample_rate
                converted_beats.append(beat_time)
            ax.axvline(
                x=beat,
                color="green",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                label="Ground Truth" if i == 0 else None,
            )
        ax.legend(loc="upper right")   
        
    if beats is not None:
        # print(f"Beats: {beats}")
        for i, beat_time in enumerate(beats):
            ax.axvline(
                x=beat_time,
                color="blue",
                linestyle="-",
                linewidth=1.0,
                alpha=0.9,
                label="Beats" if i == 0 else None,
            )
        ax.legend(loc="upper right")  
    
    if downbeats is not None:
        # print(f"Downbeats: {downbeats}")
        for i, db_time in enumerate(downbeats):
            ax.axvline(
                x=db_time,
                color="red",
                linestyle="-",
                linewidth=1.6,
                alpha=0.9,
                label="Downbeats" if i == 0 else None,
            )
        ax.legend(loc="upper right") 
    plt.savefig("beat_downbeat_plot.png", dpi=300, bbox_inches='tight')
    
def plot_log_spectrogram(log_mag_spec, ground_truth, predictions, sample_rate=44100, hop_length=441, title="Log Magnitude Spectrogram"):
    if isinstance(log_mag_spec, torch.Tensor):
        log_mag_spec = log_mag_spec.detach().cpu().numpy()

    if log_mag_spec.ndim == 3:
        log_mag_spec = log_mag_spec[0]

    fig, ax = plt.subplots(figsize=(12, 6))

    img = librosa.display.specshow(
        log_mag_spec,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax,  # <- draw on this axis
    )
    # print(len(ground_truth))
    # labels = [l for l in ground_truth if l > 0.5]  # Filter out non-zero labels
    # print(f"Batch Labels are: \n {len(labels)} \n")
    # Ground-truth lines
    converted_beats = []
    for i, beat in enumerate(ground_truth):
        if beat > 0.5: 
            beat_time = (i * hop_length) / sample_rate
            converted_beats.append(beat_time)
            ax.axvline(
                x=beat_time,
                color="green",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                label="Ground Truth" if i == 0 else None,
            )
    ax.legend(loc="upper right")   
    # print(f"Converted: {len(converted_beats)}")
    # print(f"Converted: {converted_beats}")
    # Predicted lines
    for i, beat_time in enumerate(predictions):
        ax.axvline(
            x=beat_time,
            color="blue",
            linestyle="-",
            linewidth=1.6,
            alpha=0.9,
            label="Prediction" if i == 0 else None,
        )

    ax.legend(loc="upper right")
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Log Magnitude (dB)")
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Log-Spaced Frequency Bands")
    fig.tight_layout()
    plt.show()
    
def plot_spectrogram(specgram, title=None, ylabel="freq_bin", xlabel="time", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if specgram.ndim == 3:
        specgram = specgram[0]
    specgram = specgram.detach().cpu()

    power_to_db = torchaudio.transforms.AmplitudeToDB("power", 80.0)
    ax.imshow(power_to_db(specgram).numpy(), origin="lower", aspect="auto", interpolation="nearest")
    
def plot_training_history(train_losses, valid_losses, save_path="training_history.png"):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, valid_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")
    
def plot_hmm_probabilities(activations, optimal_path, state_space, hop_length=441, sr=44100, max_time=10.0):
    acts = np.asarray(activations)
    path = np.asarray(optimal_path).ravel()

    # Support 1D (T,) and 2D (C,T) / (T,C)
    if acts.ndim == 1:
        num_frames = acts.shape[0]
        beat_probs_full = acts
        downbeat_probs_full = None
    elif acts.ndim == 2:
        # Prefer channels-first if small first dim
        if acts.shape[0] <= 4 and acts.shape[1] > acts.shape[0]:
            num_frames = acts.shape[1]
            beat_probs_full = acts[0]
            downbeat_probs_full = acts[1] if acts.shape[0] > 1 else None
        else:
            num_frames = acts.shape[0]
            beat_probs_full = acts[:, 0]
            downbeat_probs_full = acts[:, 1] if acts.shape[1] > 1 else None
    else:
        raise ValueError(f"Unsupported activations ndim={acts.ndim}, shape={acts.shape}")

    max_frame = min(int(max_time * sr / hop_length), num_frames, len(path))
    time_axis = np.arange(max_frame) * (hop_length / sr)
    optimal_path = path[:max_frame]
    beat_probs = beat_probs_full[:max_frame]
    downbeat_probs = downbeat_probs_full[:max_frame] if downbeat_probs_full is not None else None

    # 2. Calculate the 'Phase' (0.0 to 4.0) for the sawtooth plot
    phase = np.zeros(max_frame)
    for t in range(max_frame):
        curr_state = optimal_path[t]
        n = np.searchsorted(state_space.tempo_offsets, curr_state, side='right') - 1
        M = state_space.m_values[n]
        offset = state_space.tempo_offsets[n]
        pos_in_bar = curr_state - offset
        phase[t] = pos_in_bar / M

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax1.plot(time_axis, beat_probs, label='Beat Prob', color='blue', alpha=0.7)
    if downbeat_probs is not None:
        ax1.plot(time_axis, downbeat_probs, label='Downbeat Prob', color='red', alpha=0.9, linewidth=2)
    ax1.set_ylabel("Probability")
    ax1.set_title("1. TCN Observations (Input to HMM)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_axis, optimal_path, color='purple', linewidth=2)
    ax2.set_ylabel("HMM State Index")
    ax2.set_title("2. Viterbi Optimal Path (Raw States)")
    ax2.grid(True, alpha=0.3)

    ax3.plot(time_axis, phase, color='green', linewidth=2)
    for i in range(1, 4):
        ax3.axhline(y=i, color='black', linestyle='--', alpha=0.3)

    ax3.set_ylabel("Position in Bar (Beats)")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_title("3. Decoded Musical Phase (0=Downbeat, 1/2/3=Regular Beats)")
    ax3.set_yticks([0, 1, 2, 3, 4])
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
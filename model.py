import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torchaudio
# from torchaudio.transforms import Resample, Spectrogram, TimeStretch, TimeMasking, FrequencyMasking, MelScale
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
import scipy.sparse as sp
from tqdm import tqdm
from CustomPlots import plot_hmm_probabilities

# ---------------------------------------------------------------------------
# 1. Convolutional Feature Extractor
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels=1, dropout=0.1):
        super().__init__()
        # Padding (0, 1) shrinks frequency but preserves time
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=(0, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1))

        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(0, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1))

        # Collapse the remaining 8 frequency bins down to 1
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(8, 1), padding=(0, 0))

        self.activation = nn.ELU()
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Squeeze out the frequency dimension
        x = x.squeeze(2)
        return x

# ---------------------------------------------------------------------------
# 2. Temporal Convolutional Network (TCN)
# ---------------------------------------------------------------------------
class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout1d(p=dropout_rate)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout1d(p=dropout_rate)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.final_elu = nn.ELU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.elu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.elu2(out)
        out = self.dropout2(out)
        out = out + res
        out = self.final_elu(out)
        return out


class BeatTrackingTCN(nn.Module):
    def __init__(self, num_features=16, num_filters=16, kernel_size=5, dropout_rate=0.1):
        super().__init__()
        dilations = [2**i for i in range(11)]
        self.layers = nn.ModuleList([
            TCNResidualBlock(num_features, num_filters, kernel_size, d, dropout_rate)
            for d in dilations
        ])
        self.output_layer = nn.Conv1d(in_channels=num_filters, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.output_layer(x)
        out = self.sigmoid(out)
        return out

# ---------------------------------------------------------------------------
# 3. Dynamic Bayesian Network Components (State Space & Observation)
# ---------------------------------------------------------------------------
class StateSpaceModel:
    def __init__(self, min_interval=28, max_interval=109):
        self.m_values = np.arange(min_interval, max_interval + 1, dtype=np.int32)
        self.tempo_offsets = np.zeros(len(self.m_values), dtype=np.int32)
        running = 0
        for i, m in enumerate(self.m_values):
            self.tempo_offsets[i] = running
            running += int(m)
        self.total_states = int(running)

    def build_transition_matrix_log(self, p_stay=0.90, p_change=0.01):
        rows, cols, vals = [], [], []

        for n_prev, M_prev in enumerate(self.m_values):
            base_prev = int(self.tempo_offsets[n_prev])

            for m_prev in range(int(M_prev)):
                from_idx = base_prev + m_prev

                # Stay at the same tempo, move forward 1 position frame
                m_next_stay = (m_prev + 1) % M_prev
                to_idx_stay = base_prev + m_next_stay
                rows.append(from_idx)
                cols.append(to_idx_stay)
                vals.append(np.log(p_stay))

                # Tempo drifts UP (faster) to adjacent state
                if n_prev < len(self.m_values) - 1:
                    n_next_up = n_prev + 1
                    M_next_up = int(self.m_values[n_next_up])
                    m_next_up = int(round((m_prev + 1) * (M_next_up / M_prev))) % M_next_up
                    to_idx_up = int(self.tempo_offsets[n_next_up]) + m_next_up
                    
                    rows.append(from_idx)
                    cols.append(to_idx_up)
                    vals.append(np.log(p_change))

                # Tempo drifts DOWN (slower) to adjacent state
                if n_prev > 0:
                    n_next_down = n_prev - 1
                    M_next_down = int(self.m_values[n_next_down])
                    m_next_down = int(round((m_prev + 1) * (M_next_down / M_prev))) % M_next_down
                    to_idx_down = int(self.tempo_offsets[n_next_down]) + m_next_down
                    
                    rows.append(from_idx)
                    cols.append(to_idx_down)
                    vals.append(np.log(p_change))

        return sp.csr_matrix((np.array(vals, dtype=np.float64), (np.array(rows), np.array(cols))),
                             shape=(self.total_states, self.total_states))


class ObservationModel:
    def __init__(self, state_space, lambda_val=16.0):
        self.state_space = state_space
        self.lambda_val = lambda_val
        self.total_states = state_space.total_states

        self.is_beat_state = np.zeros(self.total_states, dtype=bool)
        for n, M in enumerate(state_space.m_values):
            offset = state_space.tempo_offsets[n]
            num_beat_states = max(1, int(np.round(M / self.lambda_val)))
            self.is_beat_state[offset: offset + num_beat_states] = True

    def get_observation_log_probs(self, activation_ak):
        eps = 1e-7
        ak = np.clip(activation_ak, eps, 1.0 - eps)

        obs_probs = np.zeros(self.total_states, dtype=np.float64)
        obs_probs[self.is_beat_state] = ak
        obs_probs[~self.is_beat_state] = 1.0 - ak #/ (self.lambda_val - 1.0)

        return np.log(np.clip(obs_probs, eps, None))

# ---------------------------------------------------------------------------
# 4. Viterbi Decoding Algorithm
# ---------------------------------------------------------------------------
def viterbi_decode(activations, state_space, obs_model, trans_matrix_log, hop_length=441, sr=44100):
    num_frames = len(activations)
    num_states = state_space.total_states
    A_csc = trans_matrix_log.tocsc()
    # Initialize Viterbi matrices
    v_values = np.full((num_frames, num_states), -np.inf, dtype=np.float64)
    backpointers = np.zeros((num_frames, num_states), dtype=np.int32)

    initial_log_probs = np.log(np.full(num_states, 1.0 / num_states, dtype=np.float64))
    v_values[0, :] = initial_log_probs + obs_model.get_observation_log_probs(activations[0])
    # Forward Pass of Viterbi Algorithm
    forward_pass = tqdm(range(1, num_frames), desc="Calculating Beats & Downbeats", unit="frame")
    for t in forward_pass:
        obs_log_probs = obs_model.get_observation_log_probs(activations[t])

        for j in range(num_states):
            col_start = A_csc.indptr[j]
            col_end = A_csc.indptr[j + 1]
            incoming_idx = A_csc.indices[col_start:col_end]
            incoming_probs = A_csc.data[col_start:col_end]

            if len(incoming_idx) > 0:
                path_probs = v_values[t - 1, incoming_idx] + incoming_probs
                best_prev_idx = np.argmax(path_probs)
                v_values[t, j] = path_probs[best_prev_idx] + obs_log_probs[j]
                backpointers[t, j] = incoming_idx[best_prev_idx]

    optimal_path = np.zeros(num_frames, dtype=np.int32)
    optimal_path[-1] = np.argmax(v_values[-1, :])

    for t in range(num_frames - 2, -1, -1):
        optimal_path[t] = backpointers[t + 1, optimal_path[t + 1]]

    beat_times = []
    for t in range(1, num_frames):
        current_state = optimal_path[t]
        prev_state = optimal_path[t - 1]

        tempo_idx = np.searchsorted(state_space.tempo_offsets, current_state, side='right') - 1
        position = current_state - state_space.tempo_offsets[tempo_idx]

        prev_tempo_idx = np.searchsorted(state_space.tempo_offsets, prev_state, side='right') - 1
        prev_position = prev_state - state_space.tempo_offsets[prev_tempo_idx]

        if position < prev_position: # Indicates beat transition ||| and tempo_idx == prev_tempo_idx 
            time_sec = t * (hop_length / sr)
            beat_times.append(time_sec)
    # ==================== Optional: Plot HMM Probabilities for Report ====================
    bool_hmm = False # Set to True to visualize the HMM probabilities and optimal path
    if bool_hmm:
        plot_hmm_probabilities(activations, optimal_path, state_space, hop_length, sr, max_time=10.0)
    return beat_times, optimal_path

# ---------------------------------------------------------------------------
# 5. Full Pipeline Wrapper
# ---------------------------------------------------------------------------
class BeatTrackingModel(nn.Module):
    def __init__(self, hop_length=441, sr=44100, lambda_val=16.0):
        super().__init__()
        self.conv = ConvBlock(in_channels=1, dropout=0.1) 
        self.tcn = BeatTrackingTCN(num_features=16, num_filters=16, kernel_size=5, dropout_rate=0.1)

        self.hop_length = hop_length
        self.sr = sr
        self.state_space = StateSpaceModel()
        self.obs_model = ObservationModel(self.state_space, lambda_val=lambda_val) 
        self.trans_matrix_log = self.state_space.build_transition_matrix_log()

    def forward(self, x):
        """
        Used for Training: Returns raw frame-by-frame beat probabilities (0.0 to 1.0).
        """
        x = self.conv(x)                    # (B, 16, T)
        x = self.tcn(x).squeeze(1)          # (B, T)
        return x

    def decode_with_viterbi(self, x):
        """
        Used for Inference: Returns final beat times in seconds and optimal state paths.
        """
        probs = self.forward(x)
        probs_np = probs.detach().cpu().numpy()

        beat_times_batch = []
        paths_batch = []
        
        for b in range(probs_np.shape[0]):
            beat_times, optimal_path = viterbi_decode(
                activations=probs_np[b],
                state_space=self.state_space,
                obs_model=self.obs_model,
                trans_matrix_log=self.trans_matrix_log,
                hop_length=self.hop_length,
                sr=self.sr,
            )
            beat_times_batch.append(beat_times)
            paths_batch.append(optimal_path)

        return beat_times_batch, paths_batch
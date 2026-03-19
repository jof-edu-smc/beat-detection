import os
from pathlib import Path
import torch
import torchaudio
import numpy as np
import mirdata
from torch.utils.data import Dataset

class ISMIRBeatDataset(Dataset):
    def __init__(self, pipeline, target_frames=3015):
        self.pipeline = pipeline
        self.target_frames = target_frames
        self.ballroom = self._initalize_ballroom_dataset()
        self._validate_ballroom_dataset()
        
        self.track_ids = self.ballroom.track_ids
        self.data_home = self.ballroom.default_path
        self.tracks = self.ballroom.load_tracks()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self._init_testing() 
        
    def _init_testing(self):
        # Testing section to verify dataset initialization and loading
        print(self.data_home)
        # print(self.track_ids)
        # print(self.ballroom.load_tracks())
    
    def _initalize_ballroom_dataset(self):
        working_dir = os.getcwd()
        mirdata_dir = Path(working_dir) / "mirdata"
        mirdata_dir.mkdir(parents=True, exist_ok=True)
        
        return mirdata.initialize('ballroom', data_home=str(mirdata_dir))
    
    def _validate_ballroom_dataset(self):
        valid_e = self.ballroom.validate(verbose=True)
        tracks_0 = valid_e[0]["tracks"] if isinstance(valid_e[0], dict) else valid_e[0].tracks
        tracks_1 = valid_e[1]["tracks"] if isinstance(valid_e[1], dict) else valid_e[1].tracks
        
        if tracks_0 or tracks_1:
            self.ballroom.download(force_overwrite=True, cleanup=True)  # Ensure data is downloaded
    
    def _create_target_vector(self, beat_timestamps, total_frames, hop_length, sr):
        # 1. Float32 dtype to hold decimal weights
        target = np.zeros(total_frames, dtype=np.float32)

        for beat in beat_timestamps:
            # Beat is in seconds, convert to frame index using sample rate and hop length
            frame_idx = int(round(beat * sr / hop_length))
            # print(f"Processing beat at time {beat:.3f} seconds to frame index {frame_idx} = {beat} x {sr} / {hop_length}")
            
            # 2. Assign 1.0 to the exact quantized beat location
            if 0 <= frame_idx < total_frames:
                target[frame_idx] = 1.0
            
            # 3. Widen the activation region (2 frames on either side with 0.5)
            for offset in [-2, -1, 1, 2]:
                adj_idx = frame_idx + offset
                if 0 <= adj_idx < total_frames:
                    # Use max() to ensure an overwrite of 1.0 from an adjacent beat 
                    target[adj_idx] = max(target[adj_idx], 0.5)

        return target
    
    def get_track_info(self, idx):
        key = self.track_ids[idx]
        track = self.tracks[key]
        return track.audio_path, track.beats.times, track.beats.positions, track.tempo
    
    def __len__(self):
        return len(self.track_ids)
            
    def __getitem__(self, idx):
        
        key = self.track_ids[idx]
        track = self.tracks[key]
        # print(f"Loading track {key} with audio path: {track.audio_path}")
        waveform, sr = torchaudio.load(track.audio_path)
        waveform = waveform.to(self.device)
        # Attain the single log magnitude spectrogram from the pipeline
        # Expected shape: (1, n_bands, time) -> e.g., (1, 81, time)
        log_mag_spec = self.pipeline(waveform)
        
        # Enforce fixed time length to match target_frames (e.g., 3015)
        current_t = log_mag_spec.shape[-1]
        if current_t < self.target_frames:
            # Pad the time dimension (the last dimension) with zeros
            pad_right = self.target_frames - current_t
            log_mag_spec = torch.nn.functional.pad(log_mag_spec, (0, pad_right), mode="constant", value=0.0)
        else:
            # Truncate to the target frames
            log_mag_spec = log_mag_spec[..., :self.target_frames]
        
        # Label processing: convert beat times to frame indices
        hop_length = self.pipeline.hop_length
        beats = track.beats
        beat_targets = self._create_target_vector(
            beats.times, 
            total_frames=self.target_frames, 
            hop_length=hop_length,
            sr=self.pipeline.sample_rate) # Using sample_rate from the new pipeline
        
        # Convert to float32 tensor so it acts as soft probabilities during training
        beat_targets = torch.from_numpy(beat_targets).to(torch.float32)  
        
        return log_mag_spec, beat_targets  # Returns (1, 81, 3015) and (3015,)

# Example instantiation
# full_dataset = ISMIRBeatDataset(pipeline=LogMagSpectrogramPipeline(), target_frames=3000)
# print(f"Dataset length: {len(full_dataset)}")
import os 
from STFTPipeline import LogMagSpectrogramPipeline
from model import BeatTrackingModel
import torch
import torchaudio
import numpy as np
from CustomPlots import plot_waveform
import sys

def beatTracker(inputFile, plot_predictions=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: '{device}' to process the audio and run the model.")
    # Process audio through STFT pipeline
    pipeline = LogMagSpectrogramPipeline().to(device)
    hop_length = pipeline.hop_length
    
    waveform, sr = torchaudio.load(inputFile)
    waveform = waveform.to(device)
    spectrogram = pipeline(waveform)
    # spectrogram = spectrogram.unsqueeze(0)  
    # Load model and weights
    model = BeatTrackingModel().to(device)
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        print("Loaded 'best_model.pth' successfully.")
    else:
        print("Warning: 'best_model.pth' not found. Please ensure the model weights are in the same directory")

    model.eval()
    
    # Run inference with viterbi decode
    with torch.no_grad():
        beat_times, optimal_path = model.decode_with_viterbi(
            spectrogram.unsqueeze(0).to(device=device, dtype=torch.float32)
        )
    
    downbeats = []
    # Predict tempo
    estimated_beats = np.asarray(beat_times[0], dtype=float)
    ibi_median = np.median(np.diff(estimated_beats)) if len(estimated_beats) > 1 else 0
    predicted_tempo = 60.0 / ibi_median if ibi_median > 0 else 0
    print(f"Predicted Tempo: {int(predicted_tempo)} BPM")
    path_array = np.asarray(optimal_path[0], dtype=float).ravel()
    beat_arr = np.asarray(beat_times[0], dtype=float).ravel()
    
    # tolerance for float comparison in seconds
    tol = hop_length / sr / 2.0  # half-frame tolerance
    
    # print(f"Optimal Path Length: {len(path_array)} : {path_array[:10]} with Tolerance: {tol:.4f} seconds")
    for pa in range(1, len(path_array)):  # start at 1 so pa-1 is valid
        delta = abs(path_array[pa] - path_array[pa - 1])
        
        if delta > 1.0:
            # convert current frame index to seconds
            # print(f"Optimal Path at {pa}: {path_array[pa]:.4f} | Previous: {path_array[pa - 1]:.4f} | Delta: {delta:.4f}")
            t_sec = pa * (hop_length / sr)
            seconds_per_beat = 60.0 / predicted_tempo
            seconds_per_bar = 4 * seconds_per_beat
            # check if this time matches any predicted beat time
            matches = np.where(np.isclose(beat_arr, t_sec, atol=tol))[0]
            # print(matches)
            if matches.size > 0:
                candidate_times = beat_arr[matches]
                for cand in candidate_times:
                    if (len(downbeats) == 0) or ((cand - downbeats[-1]) > seconds_per_bar):
                        downbeats.append(round(float(cand), 2))
                matches = np.array([], dtype=int)  # prevent duplicate append by the block below
                # downbeats.append(float(beat_arr[matches[0]]))
    
    # print(f"Downbeats Array: {downbeats}")
    if plot_predictions:
        plot_waveform(waveform.to('cpu'), sr=sr, beats=beat_times[0], downbeats=downbeats, title="Beat & Downbeat Detection")

    return beat_times[0], downbeats


if __name__ == "__main__":
    make_plot = False
    if len(sys.argv) > 1:
        inputFile = sys.argv[1]
        if len(sys.argv) > 2:
            make_plot = True if sys.argv[2].lower() == 'true' else False
    else:
        print("No input file provided. Please run the script like this: python main.py path/to/audio/file.wav")
        exit(0) # Default path if no argument provided
    
    beats, downbeats = beatTracker(inputFile, plot_predictions=make_plot)
    print("Detected Beats:", beats)
    print("Detected Downbeats:", downbeats)
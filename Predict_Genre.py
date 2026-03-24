
import os
import glob
import re
import unicodedata
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

# CONFIGURATION (Must match training)
SAMPLE_RATE = 16000
DURATION = 120
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 1024
FIXED_GENRES = ['Pop', 'Hip Hop', 'R&B', 'EDM', 'Indie']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# PATHS
BASE_PATH = r'c:/Users/adief/OneDrive/Dokumen/Semester 5/Data Mining 2/Week 14 (Final Project)'
AUDIO_DIR = os.path.join(BASE_PATH, 'downloads_mp3')
RESULTS_DIR = os.path.join(BASE_PATH, 'results')
VIZ_DIR = os.path.join(RESULTS_DIR, 'predictions_viz')
MODEL_PATH = os.path.join(RESULTS_DIR, 'best_model.pth')

os.makedirs(VIZ_DIR, exist_ok=True)

# --- MODEL DEFINITIONS (Must match trained model) ---
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma*out + x
        return out

class CompactCNNAttention(nn.Module):
    def __init__(self, num_classes, input_channels=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            SelfAttention(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

# --- PROCESSING UTILS ---
def load_and_preprocess_audio(file_path):
    try:
        waveform, sr = torchaudio.load(file_path)
        
        # Ensure Stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]
        
        # Resample
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Normalize Length (Center Crop or Pad to DURATION)
        max_len = SAMPLE_RATE * DURATION
        if waveform.shape[1] > max_len:
            start = (waveform.shape[1] - max_len) // 2
            waveform = waveform[:, start:start+max_len]
        else:
            pad = max_len - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad))
            
        return waveform
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_spectrogram(waveform):
    mel_transform = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    specs = []
    dummy_input = torch.zeros(1, SAMPLE_RATE * DURATION)
    expected_frames = mel_transform(dummy_input).shape[2]
    
    for i in range(2):
        spec = mel_transform(waveform[i:i+1])
        specs.append(spec)
    
    mel_spec = torch.cat(specs, dim=0)
    mel_spec = torch.log(mel_spec + 1e-6)
    
    if mel_spec.shape[2] < expected_frames:
        mel_spec = F.pad(mel_spec, (0, expected_frames - mel_spec.shape[2]))
    elif mel_spec.shape[2] > expected_frames:
        mel_spec = mel_spec[:, :, :expected_frames]
        
    return mel_spec

def visualize_prediction(filename, mel_spec, probs, pred_genre):
    plt.figure(figsize=(12, 6))
    
    # 1. Mel Spectrogram (Channel 0)
    plt.subplot(1, 2, 1)
    librosa.display.specshow(mel_spec[0].cpu().numpy(), sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram: {filename[:15]}...')
    
    # 2. Probability Bar Chart
    plt.subplot(1, 2, 2)
    colors = ['skyblue' if g != pred_genre else 'forestgreen' for g in FIXED_GENRES]
    sns.barplot(x=FIXED_GENRES, y=probs, palette=colors)
    plt.ylim(0, 1)
    plt.title(f'Prediction: {pred_genre}\nConfidence: {max(probs):.1%}')
    plt.ylabel('Confidence')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f'{filename}_pred.png'))
    plt.close()

# --- MAIN ---
if __name__ == "__main__":
    print(f"🔄 Loading model from {MODEL_PATH}...")
    model = CompactCNNAttention(num_classes=len(FIXED_GENRES))
    
    if os.path.exists(MODEL_PATH):
        # Allow loading on CPU if CUDA not available
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # Flexible model loading (handle full checkpoint or just state dict)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(DEVICE)
        model.eval()
        print("✅ Model loaded successfully!")
    else:
        print("❌ Model not found! Please train the model first.")
        exit()

    # Interactive Mode
    print("\n🎹 INTERACTIVE GENRE PREDICTION MODE")
    print("-----------------------------------")
    
    while True:
        query = input("\n📝 Enter song title or path (or 'q' to quit): ").strip()
        if query.lower() == 'q':
            break
            
        target_file = None
        
        # 1. Check if direct path
        if os.path.isfile(query):
            target_file = query
            
        # 2. Search in audio dir
        else:
            print(f"🔎 Searching for '*{query}*' in {AUDIO_DIR}...")
            # Recursive case-insensitive search
            candidates = []
            for root, _, files in os.walk(AUDIO_DIR):
                for file in files:
                    if query.lower() in file.lower() and file.lower().endswith('.mp3'):
                         candidates.append(os.path.join(root, file))
            
            if len(candidates) == 0:
                print("❌ Song not found.")
                continue
            elif len(candidates) == 1:
                target_file = candidates[0]
            else:
                print(f"⚠️  Found {len(candidates)} matches:")
                for i, c in enumerate(candidates[:10]):
                    print(f"   {i+1}. {os.path.basename(c)} ({os.path.dirname(c)})")
                if len(candidates) > 10: print("   ... and more")
                
                try:
                    choice = int(input("   Select number (0 to cancel): "))
                    if choice > 0 and choice <= len(candidates):
                        target_file = candidates[choice-1]
                    else:
                        continue
                except ValueError:
                    continue

        if target_file:
            print(f"\n🎵 Processing: {os.path.basename(target_file)}")
            
            waveform = load_and_preprocess_audio(target_file)
            if waveform is None: continue
            
            mel_spec = get_spectrogram(waveform)
            img_input = mel_spec.unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = model(img_input)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probs)
                pred_genre = FIXED_GENRES[pred_idx]
            
            print(f"✨ RESULT: {pred_genre} ({probs[pred_idx]:.1%})")
            visualize_prediction(os.path.basename(target_file), mel_spec, probs, pred_genre)
            print(f"🖼️  Saved visualization to: {VIZ_DIR}")

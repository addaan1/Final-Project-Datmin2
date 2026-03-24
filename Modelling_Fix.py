import os
import glob
import re
import unicodedata
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torchvision import models
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# CONFIG
SAMPLE_RATE = 22050
DURATION = 120
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 1024
BATCH_SIZE = 4
ACCUM_STEPS = 8
EPOCHS = 15
PATIENCE = 3
FIXED_GENRES = ['Pop', 'Hip Hop', 'R&B', 'EDM']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# PATHS
BASE_PATH = r'c:/Users/adief/OneDrive/Dokumen/Semester 5/Data Mining 2/Week 14 (Final Project)'
AUDIO_DIR = os.path.normpath(os.path.join(BASE_PATH, 'downloads_mp3'))
RESULTS_DIR = os.path.normpath(os.path.join(BASE_PATH, 'results_comparison'))
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"📁 Saving results to: {RESULTS_DIR}")

# 1. UTILS
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def normalize_text(s):
    s = str(s).lower()
    s = unicodedata.normalize('NFKD', s).encode('ascii','ignore').decode('ascii')
    s = s.replace('.mp3', ' ')
    s = re.sub(r'\[(.*?)\]|\((.*?)\)', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def get_canonical_genre(g):
    GENRE_MAP = {
        'hip hop': 'Hip Hop', 'hiphop': 'Hip Hop', 'hip-hop': 'Hip Hop', 'rap': 'Hip Hop',
        'r&b': 'R&B', 'r b': 'R&B', 'rnb': 'R&B',
        'edm': 'EDM', 'electronic': 'EDM', 'house': 'EDM', 'techno': 'EDM', 'dance': 'EDM',
        'trance': 'EDM', 'dubstep': 'EDM',
        'pop': 'Pop', 'dance pop': 'Pop',
        'rock': 'Rock', 'alternative rock': 'Rock', 'classic rock': 'Rock',
        'country': 'Country',
        'folk': 'Folk', 'acoustic': 'Folk',
        'indie': 'Indie', 'indie rock': 'Indie', 'indie pop': 'Indie',
        'soul': 'Soul', 'r&b/soul': 'Soul'
    }
    g = normalize_text(g)
    for key, val in GENRE_MAP.items():
        if key in g: return val
    return None

def load_metadata_robust(base_path):
    files = glob.glob(os.path.join(base_path, '*.csv'))
    dfs = []
    for f in files:
        for enc in ['utf-8', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(f, encoding=enc)
                df = df[['title','artist','genre']].dropna()
                df['norm_title'] = df['title'].apply(normalize_text)
                df['norm_artist'] = df['artist'].apply(normalize_text)
                df['genre_clean'] = df['genre'].apply(get_canonical_genre)
                df = df[df['genre_clean'].isin(FIXED_GENRES)]
                dfs.append(df)
                break
            except: continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def match_audio_files(meta_df, audio_dir):
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.mp3'):
                path = os.path.join(root, file).replace('\\', '/')
                audio_files.append({'path': path, 'norm_name': normalize_text(file)})
    
    matches = []
    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Matching files"):
        title = row['norm_title']
        artist = row['norm_artist']
        found = None
        for audio in audio_files:
            if title in audio['norm_name'] and artist in audio['norm_name']:
                found = audio['path']
                break
        matches.append(found)
    meta_df['file_path'] = matches
    return meta_df.dropna(subset=['file_path'])

# 2. DATASET
class StereoSpecDataset(Dataset):
    def __init__(self, df, le, train=True):
        self.df = df
        self.le = le
        self.train = train
        self.mel_transform = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        self.max_len = SAMPLE_RATE * DURATION

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            wav, sr = torchaudio.load(row['file_path'])
            # Stereo Check
            if wav.shape[0] == 1: wav = wav.repeat(2, 1)
            elif wav.shape[0] > 2: wav = wav[:2, :]
            
            if sr != SAMPLE_RATE:
                wav = T.Resample(sr, SAMPLE_RATE)(wav)
            
            # Crop/Pad
            if wav.shape[1] > self.max_len:
                start = random.randint(0, wav.shape[1] - self.max_len) if self.train else (wav.shape[1] - self.max_len)//2
                wav = wav[:, start:start+self.max_len]
            else:
                wav = F.pad(wav, (0, self.max_len - wav.shape[1]))
            
            # Spectrogram (2, 64, T)
            specs = [self.mel_transform(wav[i:i+1]) for i in range(2)]
            spec = torch.cat(specs, dim=0)
            spec = torch.log(spec + 1e-6)
            
            # Resize for Inception (if needed) but CNNs handle variable width.
            # We keep it as is (2, 64, ~1875)
            
            label = self.le.transform([row['genre_clean']])[0]
            return spec, torch.tensor(label, dtype=torch.long)
        except:
            return torch.zeros(2, N_MELS, int(self.max_len/HOP_LENGTH)+1), torch.tensor(0, dtype=torch.long)

# 3. MODEL FACTORY
def get_model(model_name, num_classes):
    print(f"🏗️  Building {model_name}...")
    
    if model_name == 'EfficientNetB0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # 1st layer: (3, 32, 3, 2) -> (2, 32, 3, 2)
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(2, old_conv.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'ResNet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # 1st layer: (3, 64, 7, 2, 3) 
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(2, old_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'MobileNetV2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # 1st layer
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(2, old_conv.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'DenseNet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        old_conv = model.features.conv0
        model.features.conv0 = nn.Conv2d(2, old_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    elif model_name == 'InceptionV3':
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.aux_logits = False # Disable aux outputs for simplicity
        model.transform_input = False # Disable internal 3-channel normalization
        # 1st layer
        old_conv = model.Conv2d_1a_3x3.conv
        model.Conv2d_1a_3x3.conv = nn.Conv2d(2, old_conv.out_channels, kernel_size=3, stride=2, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    return model.to(DEVICE)

# 4. TRAINING LOOP
def train_evaluate():
    set_seed(42)
    
    # Load Data
    print("📂 Loading Data...")
    df = load_metadata_robust(BASE_PATH)
    df = match_audio_files(df, AUDIO_DIR)
    
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['genre_clean'])
    classes = le.classes_
    print(f"🎵 Classes: {classes}")
    
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    train_ds = StereoSpecDataset(train_df, le, train=True)
    test_ds = StereoSpecDataset(test_df, le, train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    models_list = ['EfficientNetB0', 'ResNet50', 'MobileNetV2', 'DenseNet121', 'InceptionV3']
    history_log = {}
    
    for name in models_list:
        print(f"\n{'='*40}\n🚀 TRAINING {name}\n{'='*40}")
        model = get_model(name, len(classes))
        optimizer = optim.AdamW(model.parameters(), lr=1e-4) # Generic safe LR
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()
        
        train_accs, val_accs = [], []
        train_losses, val_losses = [], []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            # Train
            model.train()
            running_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            for i, (X, y) in enumerate(pbar):
                X, y = X.to(DEVICE), y.to(DEVICE)
                
                # Inception Resize trick if needed (Inception expects 299x299 usually, but lets try fully conv)
                if name == 'InceptionV3':
                    X = F.interpolate(X, size=(299, 299), mode='bilinear', align_corners=False)
                
                with autocast():
                    out = model(X)
                    loss = criterion(out, y) / ACCUM_STEPS
                
                scaler.scale(loss).backward()
                
                if (i+1) % ACCUM_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                running_loss += loss.item() * ACCUM_STEPS
                _, preds = out.max(1)
                correct += preds.eq(y).sum().item()
                total += y.size(0)
                
                pbar.set_postfix({'loss': running_loss/(i+1), 'acc': correct/total})
            
            # Eval
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    if name == 'InceptionV3':
                        X = F.interpolate(X, size=(299, 299), mode='bilinear', align_corners=False)
                    
                    with autocast():
                        out = model(X)
                        loss = criterion(out, y)
                    
                    val_loss += loss.item()
                    _, preds = out.max(1)
                    val_correct += preds.eq(y).sum().item()
                    val_total += y.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())
            
            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_acc = correct / total
            epoch_val_loss = val_loss / len(test_loader)
            epoch_val_acc = val_correct / val_total
            epoch_val_f1 = f1_score(all_targets, all_preds, average='weighted')
            
            train_losses.append(epoch_train_loss)
            train_accs.append(epoch_train_acc)
            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc)
            
            print(f"   ----------------------------------------------------------------")
            print(f"   [Epoch {epoch+1}/{EPOCHS}] Summary:")
            print(f"   🔹 Train  | Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2%}")
            print(f"   🔸 Val    | Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2%} | F1: {epoch_val_f1:.4f}")
            print(f"   ----------------------------------------------------------------")
            
            # Early Stopping Check
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f'{name}_best.pth'))
                print(f"   💾 Best model saved (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"   🛑 Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Save History & Plots
        history_log[name] = {
            'train_acc': train_accs, 'val_acc': val_accs,
            'train_loss': train_losses, 'val_loss': val_losses,
            'final_acc': val_accs[-1],
            'preds': all_preds, 'targets': all_targets
        }
        
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f'{name}_final.pth'))
        
        # Classification Report
        report = classification_report(all_targets, all_preds, target_names=classes)
        print(f"\n📊 Classification Report for {name}:\n{report}")
        with open(os.path.join(RESULTS_DIR, 'classification_reports.txt'), 'a') as f:
            f.write(f"\n{'='*40}\nModel: {name}\n{'='*40}\n{report}\n")

    return history_log, classes

# 5. VISUALIZATION
def create_visualizations(history, classes):
    print("\n📊 Generating Report...")
    
    # 1. Comparison Bar Chart
    names = list(history.keys())
    accs = [history[n]['final_acc'] for n in names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accs, color=sns.color_palette("viridis", len(names)))
    plt.title("Model Comparison - Final Test Accuracy")
    plt.ylim(0, 1.0)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1%}', ha='center', va='bottom')
    plt.savefig(os.path.join(RESULTS_DIR, 'model_accuracy_comparison.png'))
    plt.close()
    
    # 2. Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, name in enumerate(names):
        cm = confusion_matrix(history[name]['targets'], history[name]['preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], xticklabels=classes, yticklabels=classes)
        axes[i].set_title(f"{name} (Acc: {history[name]['final_acc']:.1%})")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
        
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrices.png'))
    plt.close()
    
    # 3. Training Curves
    plt.figure(figsize=(12, 5))
    for name in names:
        plt.plot(history[name]['val_acc'], label=name)
    plt.title("Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'))
    plt.close()
    
    print(f"✅ Comparison complete! Check {RESULTS_DIR}")

if __name__ == "__main__":
    history, classes = train_evaluate()
    create_visualizations(history, classes)

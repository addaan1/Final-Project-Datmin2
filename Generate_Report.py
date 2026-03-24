
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F

# Import necessary components from Modelling_Fix
# We assume Modelling_Fix is in the same directory and verified to work
from Modelling_Fix import (
    load_metadata_robust, match_audio_files, StereoSpecDataset, get_model,
    set_seed, normalize_text, get_canonical_genre,
    BASE_PATH, AUDIO_DIR, RESULTS_DIR, DEVICE, BATCH_SIZE, EPOCHS, FIXED_GENRES,
    train_test_split, LabelEncoder, T, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, DURATION
)

def generate_best_reports():
    print("🔁 Re-loading Test Data (Exact Split)...")
    set_seed(42) # CRITICAL: Must match training seed
    
    # Reload Data
    df = load_metadata_robust(BASE_PATH)
    df = match_audio_files(df, AUDIO_DIR)
    
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['genre_clean'])
    classes = le.classes_
    print(f"🎵 Classes: {classes}")
    
    # Split
    _, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    test_ds = StereoSpecDataset(test_df, le, train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    models_list = ['EfficientNetB0', 'ResNet50', 'MobileNetV2', 'DenseNet121', 'InceptionV3']
    results = {}
    
    print("\n🚀 Starting Evaluation of BEST Models...")
    
    final_report_path = os.path.join(RESULTS_DIR, 'classification_reports_best.txt')
    if os.path.exists(final_report_path): os.remove(final_report_path)
        
    for name in models_list:
        model_path = os.path.join(RESULTS_DIR, f'{name}_best.pth')
        if not os.path.exists(model_path):
            print(f"❌ {name}: Best model not found at {model_path}. Skipping.")
            continue
            
        print(f"📥 Loading {name} (Best)...")
        model = get_model(name, len(classes))
        
        # Load Weights
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X, y in tqdm(test_loader, desc=f"Evaluating {name}"):
                X, y = X.to(DEVICE), y.to(DEVICE)
                
                if name == 'InceptionV3':
                    X = F.interpolate(X, size=(299, 299), mode='bilinear', align_corners=False)
                    
                with autocast():
                    out = model(X)
                
                _, preds = out.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        # Metrics
        acc = accuracy_score(all_targets, all_preds)
        report = classification_report(all_targets, all_preds, target_names=classes)
        results[name] = {'acc': acc, 'preds': all_preds, 'targets': all_targets}
        
        print(f"✅ {name}: Accuracy = {acc:.2%}")
        print(report) # Print report to console
        
        # Write Report
        with open(final_report_path, 'a') as f:
            f.write(f"\n{'='*40}\nModel: {name} (BEST)\n{'='*40}\n")
            f.write(report)
            f.write(f"Accuracy: {acc:.4f}\n\n")

    # Re-generate Visualizations
    if results:
        curr_viz_dir = os.path.join(RESULTS_DIR, 'viz_best')
        os.makedirs(curr_viz_dir, exist_ok=True)
        print(f"\n📊 Generating Visualizations in {curr_viz_dir}...")
        
        # 1. Bar Chart
        names = list(results.keys())
        accs = [results[n]['acc'] for n in names]
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, accs, color=sns.color_palette("viridis", len(names)))
        plt.title("Best Model Accuracy Comparison")
        plt.ylim(0, 1.0)
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1%}', ha='center', va='bottom')
        plt.savefig(os.path.join(curr_viz_dir, 'accuracy_comparison_best.png'))
        plt.close()
        
        # 2. Confusion Matrices
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        for i, name in enumerate(names):
            cm = confusion_matrix(results[name]['targets'], results[name]['preds'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], xticklabels=classes, yticklabels=classes)
            axes[i].set_title(f"{name}\n(Acc: {results[name]['acc']:.1%})")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(curr_viz_dir, 'confusion_matrices_best.png'))
        plt.close()
        
        print(f"🎉 Done! Reports saved to {final_report_path}")

if __name__ == "__main__":
    generate_best_reports()

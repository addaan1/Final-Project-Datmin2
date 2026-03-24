<div align="center">

# Klasifikasi Genre Musik Indonesia (2016-2025)
### Transfer Learning pada CNN Berbasis Spektrogram Audio

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Academic-blueviolet?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)]()

<br/>

**Final Project - Data Mining 2 | Kelompok 8**

| NIM | Nama |
|:---:|:---|
| 164231013 | Sahrul Adicandra Effendy |
| 164231052 | Cuthbert Young |
| 164231061 | Evan Nathaniel Susanto |
| 164231077 | Atilla Verel Arrizqi |
| 164231095 | Mohammad Faizal Aprilianto |

---

</div>

## Daftar Isi

- [Abstract](#abstract)
- - [Latar Belakang](#latar-belakang)
  - - [Metodologi](#metodologi)
    - - [Hasil and Performa Model](#hasil-and-performa-model)
      - - [Analisis Mendalam](#analisis-mendalam)
        - - [Struktur Proyek](#struktur-proyek)
          - - [Konfigurasi Eksperimen](#konfigurasi-eksperimen)
            - - [Cara Penggunaan](#cara-penggunaan)
              - - [Dataset and Model](#dataset-and-model)
                - - [Referensi Kunci](#referensi-kunci)
                 
                  - ---

                  ## Abstract

                  Penelitian ini mengembangkan sistem **klasifikasi genre musik Indonesia** menggunakan representasi **Mel-Spectrogram** yang dipadukan dengan lima arsitektur **Convolutional Neural Networks (CNN)** melalui pendekatan *transfer learning*. Dataset mencakup lagu-lagu Indonesia tahun **2016-2025** yang difokuskan pada empat genre utama: **Pop, Hip Hop, R&B, dan EDM**.

                  Hasil eksperimen menunjukkan **EfficientNetB0** sebagai arsitektur terbaik dengan **F1-Score 0.73** dan akurasi validasi **74.5%**, melampaui DenseNet121, MobileNetV2, ResNet50, dan InceptionV3. Penelitian ini juga mengungkap tantangan signifikan akibat *class imbalance* pada dataset musik lokal Indonesia.

                  **Keywords:** "music genre classification", "Mel-Spectrogram", "CNN", "transfer learning", "EfficientNetB0", "class imbalance"

                  ---

                  ## Latar Belakang

                  Pertumbuhan pesat platform musik digital (Spotify, YouTube, Apple Music) menghasilkan volume data audio yang masif, sekaligus menuntut sistem klasifikasi genre yang akurat. Penelitian ini berfokus pada:

                  - **Tantangan utama:** Ketidakseimbangan kelas (class imbalance) yang ekstrem pada dataset musik Indonesia
                  - - **Pendekatan:** Komparasi komprehensif 5 arsitektur CNN modern dengan transfer learning berbasis representasi Mel-Spectrogram
                    - - **Konteks lokal:** Evaluasi spesifik pada karakteristik musik Indonesia yang heterogen periode 2016-2025
                     
                      - ---

                      ## Metodologi

                      ### Pipeline Lengkap

                      ```
                      Data Collection -> Metadata Preprocessing -> Audio Preprocessing
                            |                    |                       |
                            v                    v                       v
                      Apple Music           Normalisasi             WAV Format
                      Spotify            Genre Mapping:          22.050 Hz Stereo
                      LastFM          Pop / Hip Hop / R&B / EDM    120 detik
                      YouTube                                    (Fixed Duration)
                            |
                            v
                      Feature Extraction: Mel-Spectrogram (64 Mel bands, Log-dB scale)
                      Tensor: (2, 64, Time)
                            |
                            v
                      Transfer Learning CNN Training
                      +-----------------------------------------+
                      |  EfficientNetB0  |  ResNet50            |
                      |  MobileNetV2     |  DenseNet121         |
                      |  InceptionV3                            |
                      |                                         |
                      |  AdamW Optimizer (lr=1e-4)             |
                      |  Stratified Split 80:20                 |
                      |  Mixed Precision Training (AMP)         |
                      |  Early Stopping (patience=3)            |
                      |  Max 15 Epoch                          |
                      +-----------------------------------------+
                            |
                            v
                      Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
                      ```

                      ### Tahap Pra-Pemrosesan Audio

                      | Tahap | Parameter | Keterangan |
                      |:------|:----------|:-----------|
                      | Format Audio | WAV | Konversi dari berbagai format sumber |
                      | Sample Rate | 22.050 Hz | Standar analisis musik |
                      | Channel | Stereo (2-ch) | Mono diduplikasi, >2ch dipotong |
                      | Durasi | 120 detik (tetap) | Train: random segment, Val/Test: segment tengah |
                      | Mel Bands | 64 | Representasi frekuensi skala Mel |
                      | Skala | Logaritmik (dB) | Meniru sensitivitas pendengaran manusia |
                      | Output Tensor | (2, 64, T) | Input ke arsitektur CNN |

                      ---

                      ## Hasil and Performa Model

                      ### Tabel Komparasi Model

                      | Peringkat | Arsitektur | Precision | Recall | **F1-Score** | Val. Accuracy |
                      |:---------:|:-----------|:---------:|:------:|:------------:|:-------------:|
                      | 1 | **EfficientNetB0** | **0.73** | **0.75** | **0.73** | **74.5%** |
                      | 2 | DenseNet121 | 0.66 | 0.73 | 0.69 | 72.6% |
                      | 3 | MobileNetV2 | 0.62 | 0.71 | 0.65 | 70.8% |
                      | 4 | InceptionV3 | 0.60 | 0.68 | 0.63 | 67.9% |
                      | 5 | ResNet50 | 0.58 | 0.69 | 0.62 | 68.9% |

                      ### Visualisasi Performa

                      **Perbandingan Akurasi Validasi Terbaik (5 Model)**

                      ![Model Accuracy Comparison](results_comparison/model_accuracy_comparison.png)

                      **Kurva Pelatihan (Training Curves)**

                      ![Training Curves](results_comparison/training_curves.png)

                      **Confusion Matrix (5 Arsitektur)**

                      ![Confusion Matrices](results_comparison/confusion_matrices.png)

                      ---

                      ## Analisis Mendalam

                      ### Kenapa EfficientNetB0 Unggul?

                      EfficientNetB0 menerapkan prinsip compound scaling - menyeimbangkan kedalaman, lebar, dan resolusi jaringan secara simultan. Dengan hanya 5.3 juta parameter (vs. ResNet50 dengan 25M+), model ini lebih efisien dan tidak mudah overfit pada dataset berukuran sedang.

                      ### Dampak Class Imbalance

                      ```
                      Distribusi Genre dalam Dataset:
                      Pop      ############################-- (Mayoritas Ekstrem)
                      Hip Hop  ###############---------------
                      R&B      #####------------------------- (Minoritas)
                      EDM      ####-------------------------- (Minoritas)
                      ```

                      - **R&B** adalah genre paling sulit diprediksi karena jumlah sampel sangat sedikit
                      - - **EDM**: EfficientNetB0 berhasil prediksi 7 sampel EDM dengan benar, ResNet50 gagal total (0 benar)
                        - - **Pop <-> Hip Hop**: Terjadi kebingungan pada semua model akibat irisan fitur spektral
                         
                          - ### Studi Kasus: Prediksi Nyata
                         
                          - **Klasifikasi Berhasil** - "Lil Pump - Gucci Gang" (Ground Truth: Hip Hop)
                         
                          - | Model | Prediksi | Confidence |
                          - |:------|:---------|:----------:|
                          - | **EfficientNetB0** | Hip Hop | **97.9%** |
                          - | MobileNetV2 | Hip Hop | 97.9% |
                          - | DenseNet121 | Hip Hop | 85.1% |
                          - | InceptionV3 | Hip Hop | 79.7% |
                          - | ResNet50 | Hip Hop | 67.2% |
                         
                          - **Misklasifikasi Kolektif** - "Ed Sheeran - Shape Of You" (Ground Truth: Pop)
                         
                          - | Model | Prediksi | Confidence |
                          - |:------|:---------|:----------:|
                          - | **EfficientNetB0** | R&B | 65.5% (Pop: 28.3%) |
                          - | MobileNetV2 | R&B | 77.8% (Pop: 13.9%) |
                          - | DenseNet121 | R&B | 74.5% |
                          - | InceptionV3 | R&B | 55.6% |
                          - | ResNet50 | R&B | 58.9% |
                         
                          - ---

                          ## Konfigurasi Eksperimen

                          ```python
                          CONFIG = {
                              "optimizer":        "AdamW",
                              "learning_rate":    1e-4,
                              "batch_size":       4,          # Effective batch = 32 (via gradient accumulation)
                              "accum_steps":      8,
                              "max_epochs":       15,
                              "early_stopping":   3,          # Patience
                              "loss_function":    "CrossEntropyLoss",
                              "precision":        "Mixed (AMP + GradScaler)",
                              "split_strategy":   "Stratified 80:20",
                              "mel_bands":        64,
                              "sample_rate":      22050,
                              "audio_duration":   120,
                          }
                          ```

                          ---

                          ## Struktur Proyek

                          ```
                          Final-Project-Datmin2/
                          |
                          |-- DM2_A2_013_052_061_077_095.ipynb
                          |-- Modelling.py
                          |-- Modelling_Fix.py
                          |-- Predict_Genre.py
                          |-- Generate_Report.py
                          |-- Flowchart.Rmd
                          |-- 2016.csv - 2025.csv
                          |
                          |-- results/
                          |   `-- model weights per arsitektur
                          |
                          `-- results_comparison/
                              |-- confusion_matrices.png
                              |-- model_accuracy_comparison.png
                              |-- training_curves.png
                              |-- classification_reports.txt
                              `-- *_best.pth / *_final.pth
                          ```

                          ---

                          ## Cara Penggunaan

                          ```bash
                          # Install dependencies
                          pip install torch torchvision torchaudio librosa numpy pandas scikit-learn matplotlib seaborn

                          # Prediksi genre dari file audio
                          python Predict_Genre.py --audio "path/to/song.mp3" --model efficientnetb0

                          # Training dari awal
                          python Modelling_Fix.py

                          # Generate laporan evaluasi
                          python Generate_Report.py
                          ```

                          ---

                          ## Dataset and Model

                          | Resource | Link |
                          |:---------|:----:|
                          | Dataset Lagu Indonesia | [Google Drive](https://drive.google.com/drive/u/0/folders/1ldNJaW2lq5XYQMmqRE-sW2N7WopAvbkw) |
                          | Pre-trained Best Models | [Google Drive](https://drive.google.com/drive/u/1/folders/1sBRK8eQbu1MWOS27pDV0d95N0MF6QTif) |

                          ---

                          ## Pengembangan Selanjutnya

                          | Area | Teknik | Manfaat |
                          |:-----|:-------|:--------|
                          | Loss Function | Focal Loss | Mengurangi bias kelas mayoritas |
                          | Augmentasi Data | SpecAugment | Perkaya variasi spektrogram latih |
                          | Arsitektur | Audio Spectrogram Transformer (AST) | Tangkap dependensi temporal panjang |
                          | Arsitektur Hibrida | CNN-RNN / CRNN | Gabungkan fitur spasial and temporal |

                          ---

                          ## Referensi Kunci

                          - Tan, M., and Le, Q. V. (2019). EfficientNet. ICML 2019.
                          - - Gong, Y., et al. (2021). AST: Audio Spectrogram Transformer. Interspeech 2021.
                            - - Park, D. S., et al. (2019). SpecAugment. Interspeech 2019.
                              - - Rosmala, D., and Fadhilah, M. N. (2025). Audio Conversion for Music Genre Classification. Elkomika.
                                - - Ashraf, M., et al. (2023). Hybrid CNN and RNN for music classification. Applied Sciences.
                                 
                                  - ---

                                  <div align="center">

                                  **Kode:** DM2_A2_013_052_061_077_095

                                  *Final Project Data Mining 2 - Semester 5, 2025*

                                  Jika bermanfaat, berikan bintang!

                                  </div>
                                  

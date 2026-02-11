# whisper-finetune-lora

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)
![Whisper large-v3](https://img.shields.io/badge/model-whisper--large--v3-orange)

Fine-tune OpenAI Whisper large-v3 with LoRA (FP16) for any language, then convert to faster-whisper format for production inference.

Whisper large-v3'u LoRA (FP16) ile herhangi bir dil icin fine-tune edin, ardindan production inference icin faster-whisper formatina donusturun.

---

> **EN** | [Quick Start](#quick-start) | [How It Works](#how-it-works) | [Supported Datasets](#supported-datasets) | [Configuration](#configuration) | [GPU Requirements](#gpu-requirements) | [License](#license)
>
> **TR** | [Hizli Baslangic](#hizli-baslangic) | [Nasil Calisir](#nasil-calisir) | [Desteklenen Veri Setleri](#desteklenen-veri-setleri) | [Yapilandirma](#yapilandirma) | [GPU Gereksinimleri](#gpu-gereksinimleri) | [Lisans](#lisans-1)

---

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 24GB+ VRAM (A100, A30, RTX 4090, etc.)
- CUDA 11.8+ and cuDNN
- HuggingFace account with token (`huggingface-cli login`)

### Steps

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/whisper-finetune-lora.git
cd whisper-finetune-lora
pip install -r requirements.txt

# 2. Login to HuggingFace (required for dataset downloads)
huggingface-cli login

# 3. See available languages
python prepare_data.py --list-languages

# 4. Run full pipeline for your language
bash run_all.sh tr          # Turkish
bash run_all.sh de          # German
bash run_all.sh en          # English

# 5. Your model is ready at ./output/{lang}/faster-whisper-{lang}/
```

### Step by Step (Manual)

```bash
# Download and preprocess data
python prepare_data.py --language tr --config config.yaml

# Train with LoRA
python train_lora.py --language tr --config config.yaml

# Merge LoRA weights and convert to faster-whisper
python merge_and_convert.py --language tr --config config.yaml

# Evaluate (with optional baseline comparison)
python evaluate.py --language tr --config config.yaml --compare-baseline
```

### Use the Fine-tuned Model

```python
from faster_whisper import WhisperModel

model = WhisperModel("./output/tr/faster-whisper-tr", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.wav")
for segment in segments:
    print(f"[{segment.start:.2f} -> {segment.end:.2f}] {segment.text}")
```

## How It Works

```
HuggingFace Datasets  ──>  prepare_data.py  ──>  Preprocessed Data
                                                        │
                                                        v
                                                  train_lora.py
                                                        │
                                               LoRA FP16 Checkpoints
                                                        │
                                                        v
                                              merge_and_convert.py
                                                        │
                                            CTranslate2 / faster-whisper
                                                        │
                                                        v
                                                   evaluate.py
                                                        │
                                               WER / CER / RTFx Report
```

**Pipeline:**

1. **Data Preparation**: Downloads datasets for the target language from multiple sources (Common Voice, FLEURS, VoxPopuli, MLS, ISSAI). Normalizes columns, resamples audio, applies optional augmentation.
2. **LoRA Training**: Applies Low-Rank Adaptation to Whisper's attention layers (`q_proj`, `v_proj`) and feed-forward (`fc1`). Trains with FP16, early stopping, and WER-based model selection.
3. **Merge & Convert**: Merges LoRA weights back into the base model, then converts to CTranslate2 format for faster-whisper inference.
4. **Evaluation**: Measures Word Error Rate (WER), Character Error Rate (CER), and Real-Time Factor (RTFx) on test data. Optional baseline comparison with the original large-v3.

## Supported Datasets

The script automatically selects datasets that contain the target language:

| Dataset | Languages | License | Attribution |
|---------|-----------|---------|-------------|
| [Common Voice 17.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | 121 | CC-0 | - |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | 102 | CC-BY-4.0 | Google Research |
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) | 18 | CC0-1.0 | - |
| [MLS](https://huggingface.co/datasets/facebook/multilingual_librispeech) | 8 | CC-BY-4.0 | Meta AI |
| [ISSAI Turkish](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus) | 1 (tr) | MIT | ISSAI |

Example: `--language tr` pulls from Common Voice + FLEURS + ISSAI (3 sources, ~230+ hours).

Run `python prepare_data.py --list-languages` to see all languages and their available sources.

## Configuration

All hyperparameters are in `config.yaml`. Language is specified via CLI `--language` flag.

Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora.r` | 32 | LoRA rank |
| `lora.alpha` | 64 | LoRA alpha |
| `lora.target_modules` | q_proj, v_proj, fc1 | Adapted layers |
| `training.learning_rate` | 1e-4 | Learning rate |
| `training.batch_size` | 8 | Per-device batch size |
| `training.num_epochs` | 10 | Max epochs (early stopping enabled) |
| `output.ct2_quantization` | float16 | CTranslate2 output precision |

## Output Structure

```
output/
└── tr/
    ├── data/           # Preprocessed train/val/test splits
    ├── checkpoints/    # Training checkpoints + best model
    ├── merged/         # HuggingFace format merged model
    ├── faster-whisper-tr/  # Production-ready CTranslate2 model
    └── results.json    # WER, CER, RTFx evaluation results
```

## GPU Requirements

| GPU | VRAM | Batch Size | Notes |
|-----|------|------------|-------|
| A100 40GB | 40 GB | 16 | Recommended |
| A30 24GB | 24 GB | 8 | Default config |
| RTX 4090 | 24 GB | 8 | Good alternative |
| RTX 3090 | 24 GB | 8 | Works |
| T4 16GB | 16 GB | 4 | Reduce batch_size |

For lower VRAM GPUs, reduce `training.batch_size` and increase `training.gradient_accumulation_steps` in `config.yaml`.

## Cost Estimate (Turkish Example)

| Item | Cost |
|------|------|
| Cloud GPU A100 40GB (~30-60 hrs) | $20-40 |
| Data download | $0 (open source) |
| **Total** | **$20-40** |

See [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) for GPU provider comparisons.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

### Training Data Attribution

- **FLEURS**: Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech" (CC-BY-4.0, Google Research)
- **MLS**: Pratap et al., "MLS: A Large-Scale Multilingual Dataset for Speech Research" (CC-BY-4.0, Meta AI)

### Dependencies

- [OpenAI Whisper](https://github.com/openai/whisper) (MIT)
- [PEFT](https://github.com/huggingface/peft) (Apache 2.0)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) (MIT)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT)

---

## Hizli Baslangic

### On Kosullar

- Python 3.10+
- 24GB+ VRAM'a sahip NVIDIA GPU (A100, A30, RTX 4090, vb.)
- CUDA 11.8+ ve cuDNN
- HuggingFace hesabi ve token (`huggingface-cli login`)

### Adimlar

```bash
# 1. Klonla ve kur
git clone https://github.com/YOUR_USERNAME/whisper-finetune-lora.git
cd whisper-finetune-lora
pip install -r requirements.txt

# 2. HuggingFace'e giris yap (veri seti indirmek icin gerekli)
huggingface-cli login

# 3. Mevcut dilleri gor
python prepare_data.py --list-languages

# 4. Sectigin dil icin tum pipeline'i calistir
bash run_all.sh tr          # Turkce
bash run_all.sh de          # Almanca
bash run_all.sh en          # Ingilizce

# 5. Modelin hazir: ./output/{dil}/faster-whisper-{dil}/
```

### Fine-tune Edilmis Modeli Kullan

```python
from faster_whisper import WhisperModel

model = WhisperModel("./output/tr/faster-whisper-tr", device="cuda", compute_type="float16")
segments, info = model.transcribe("ses.wav")
for segment in segments:
    print(f"[{segment.start:.2f} -> {segment.end:.2f}] {segment.text}")
```

## Nasil Calisir

1. **Veri Hazirlama**: Hedef dil icin birden fazla kaynaktan (Common Voice, FLEURS, VoxPopuli, MLS, ISSAI) veri setlerini indirir. Kolonlari normalize eder, sesi yeniden ornekler, opsiyonel augmentation uygular.
2. **LoRA Egitimi**: Whisper'in attention katmanlarina (`q_proj`, `v_proj`) ve feed-forward katmanina (`fc1`) Low-Rank Adaptation uygular. FP16 ile egitir, early stopping ve WER-tabanli model secimi yapar.
3. **Birlestirme & Donusum**: LoRA agirliklarini temel modele geri birlestirir, ardindan faster-whisper inference icin CTranslate2 formatina donusturur.
4. **Degerlendirme**: Test verisi uzerinde WER, CER ve RTFx olcer. Opsiyonel olarak orijinal large-v3 ile karsilastirir.

## Desteklenen Veri Setleri

Script, hedef dili iceren veri setlerini otomatik olarak secer:

| Veri Seti | Dil Sayisi | Lisans | Atif |
|-----------|------------|--------|------|
| [Common Voice 17.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | 121 | CC-0 | - |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | 102 | CC-BY-4.0 | Google Research |
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) | 18 | CC0-1.0 | - |
| [MLS](https://huggingface.co/datasets/facebook/multilingual_librispeech) | 8 | CC-BY-4.0 | Meta AI |
| [ISSAI Turkish](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus) | 1 (tr) | MIT | ISSAI |

Ornek: `--language tr` Common Voice + FLEURS + ISSAI'den (3 kaynak, 230+ saat) veri ceker.

Tum dilleri ve kaynaklarini gormek icin: `python prepare_data.py --list-languages`

## Yapilandirma

Tum hyperparameter'lar `config.yaml` dosyasindadir. Dil, CLI'da `--language` ile belirtilir.

## GPU Gereksinimleri

| GPU | VRAM | Batch Boyutu | Not |
|-----|------|-------------|-----|
| A100 40GB | 40 GB | 16 | Onerilen |
| A30 24GB | 24 GB | 8 | Varsayilan ayar |
| RTX 4090 | 24 GB | 8 | Iyi alternatif |
| RTX 3090 | 24 GB | 8 | Calisir |
| T4 16GB | 16 GB | 4 | batch_size azaltilmali |

Dusuk VRAM'li GPU'lar icin `config.yaml`'da `training.batch_size` degerini azaltip `training.gradient_accumulation_steps` degerini artirin.

## Maliyet Tahmini (Turkce Ornegi)

| Kalem | Maliyet |
|-------|---------|
| Bulut GPU A100 40GB (~30-60 saat) | $20-40 |
| Veri indirme | $0 (acik kaynak) |
| **Toplam** | **$20-40** |

GPU saglayici karsilastirmasi icin [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) dosyasina bakin.

## Lisans

Bu proje [Apache License 2.0](LICENSE) altinda lisanslanmistir.

### Egitim Verisi Atfi

- **FLEURS**: Conneau ve ark., "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech" (CC-BY-4.0, Google Research)
- **MLS**: Pratap ve ark., "MLS: A Large-Scale Multilingual Dataset for Speech Research" (CC-BY-4.0, Meta AI)

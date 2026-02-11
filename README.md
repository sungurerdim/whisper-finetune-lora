# whisper-finetune-lora

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Whisper large-v3](https://img.shields.io/badge/model-whisper--large--v3-orange)

Fine-tune OpenAI Whisper large-v3 with LoRA (FP16) for any language, then convert to faster-whisper format for production inference.

Whisper large-v3'ü LoRA (FP16) ile herhangi bir dil için fine-tune edin, ardından production inference için faster-whisper formatına dönüştürün.

---

> **EN** | [Quick Start](#quick-start) | [How It Works](#how-it-works) | [Supported Datasets](#supported-datasets) | [Configuration](#configuration) | [GPU Requirements](#gpu-requirements) | [License](#license)
>
> **TR** | [Hızlı Başlangıç](#hızlı-başlangıç) | [Nasıl Çalışır](#nasıl-çalışır) | [Desteklenen Veri Setleri](#desteklenen-veri-setleri) | [Yapılandırma](#yapılandırma) | [GPU Gereksinimleri](#gpu-gereksinimleri) | [Lisans](#lisans-1)

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
git clone https://github.com/sungurerdim/whisper-finetune-lora.git
cd whisper-finetune-lora
pip install -r requirements.txt

# 2. Login to HuggingFace (required for dataset downloads)
huggingface-cli login

# 3. See available languages
python prepare_data.py --list-languages

# 4. Run the interactive CLI
python cli.py

# Or run full pipeline directly for your language
bash run_all.sh tr          # Turkish
bash run_all.sh de          # German
bash run_all.sh en          # English

# 5. Your model is ready at ./output/{lang}/faster-whisper-{lang}/
```

### Interactive CLI

The easiest way to use the pipeline. Provides language search, config display, and step-by-step guidance:

```bash
python cli.py
```

```
┌── Whisper LoRA Fine-tuning ──────────┐
│                                      │
│  [1]  Veri Hazirla                   │
│  [2]  LoRA Egitimi                   │
│  [3]  Birlestir & Donustur           │
│  [4]  Degerlendir                    │
│  [5]  Tum Pipeline'i Calistir        │
│  [6]  Yapilandirmayi Goster          │
│  [0]  Cikis                          │
│                                      │
└──────────────────────────────────────┘
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

This project is licensed under the [MIT License](LICENSE).

### Training Data Attribution

- **FLEURS**: Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech" (CC-BY-4.0, Google Research)
- **MLS**: Pratap et al., "MLS: A Large-Scale Multilingual Dataset for Speech Research" (CC-BY-4.0, Meta AI)

### Dependencies

- [OpenAI Whisper](https://github.com/openai/whisper) (MIT)
- [PEFT](https://github.com/huggingface/peft) (Apache 2.0)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) (MIT)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT)

---

## Hızlı Başlangıç

### Ön Koşullar

- Python 3.10+
- 24GB+ VRAM'a sahip NVIDIA GPU (A100, A30, RTX 4090, vb.)
- CUDA 11.8+ ve cuDNN
- HuggingFace hesabı ve token (`huggingface-cli login`)

### Adımlar

```bash
# 1. Klonla ve kur
git clone https://github.com/sungurerdim/whisper-finetune-lora.git
cd whisper-finetune-lora
pip install -r requirements.txt

# 2. HuggingFace'e giriş yap (veri seti indirmek için gerekli)
huggingface-cli login

# 3. Mevcut dilleri gör
python prepare_data.py --list-languages

# 4. İnteraktif CLI'ı çalıştır
python cli.py

# Veya doğrudan tüm pipeline'ı çalıştır
bash run_all.sh tr          # Türkçe
bash run_all.sh de          # Almanca
bash run_all.sh en          # İngilizce

# 5. Modelin hazır: ./output/{dil}/faster-whisper-{dil}/
```

### Fine-tune Edilmiş Modeli Kullan

```python
from faster_whisper import WhisperModel

model = WhisperModel("./output/tr/faster-whisper-tr", device="cuda", compute_type="float16")
segments, info = model.transcribe("ses.wav")
for segment in segments:
    print(f"[{segment.start:.2f} -> {segment.end:.2f}] {segment.text}")
```

## Nasıl Çalışır

1. **Veri Hazırlama**: Hedef dil için birden fazla kaynaktan (Common Voice, FLEURS, VoxPopuli, MLS, ISSAI) veri setlerini indirir. Kolonları normalize eder, sesi yeniden örnekler, opsiyonel augmentation uygular.
2. **LoRA Eğitimi**: Whisper'ın attention katmanlarına (`q_proj`, `v_proj`) ve feed-forward katmanına (`fc1`) Low-Rank Adaptation uygular. FP16 ile eğitir, early stopping ve WER-tabanlı model seçimi yapar.
3. **Birleştirme & Dönüşüm**: LoRA ağırlıklarını temel modele geri birleştirir, ardından faster-whisper inference için CTranslate2 formatına dönüştürür.
4. **Değerlendirme**: Test verisi üzerinde WER, CER ve RTFx ölçer. Opsiyonel olarak orijinal large-v3 ile karşılaştırır.

## Desteklenen Veri Setleri

Script, hedef dili içeren veri setlerini otomatik olarak seçer:

| Veri Seti | Dil Sayısı | Lisans | Atıf |
|-----------|------------|--------|------|
| [Common Voice 17.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | 121 | CC-0 | - |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | 102 | CC-BY-4.0 | Google Research |
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) | 18 | CC0-1.0 | - |
| [MLS](https://huggingface.co/datasets/facebook/multilingual_librispeech) | 8 | CC-BY-4.0 | Meta AI |
| [ISSAI Turkish](https://huggingface.co/datasets/issai/Turkish_Speech_Corpus) | 1 (tr) | MIT | ISSAI |

Örnek: `--language tr` Common Voice + FLEURS + ISSAI'den (3 kaynak, 230+ saat) veri çeker.

Tüm dilleri ve kaynaklarını görmek için: `python prepare_data.py --list-languages`

## Yapılandırma

Tüm hyperparameter'lar `config.yaml` dosyasındadır. Dil, CLI'da `--language` ile belirtilir.

## GPU Gereksinimleri

| GPU | VRAM | Batch Boyutu | Not |
|-----|------|-------------|-----|
| A100 40GB | 40 GB | 16 | Önerilen |
| A30 24GB | 24 GB | 8 | Varsayılan ayar |
| RTX 4090 | 24 GB | 8 | İyi alternatif |
| RTX 3090 | 24 GB | 8 | Çalışır |
| T4 16GB | 16 GB | 4 | batch_size azaltılmalı |

Düşük VRAM'li GPU'lar için `config.yaml`'da `training.batch_size` değerini azaltıp `training.gradient_accumulation_steps` değerini artırın.

## Maliyet Tahmini (Türkçe Örneği)

| Kalem | Maliyet |
|-------|---------|
| Bulut GPU A100 40GB (~30-60 saat) | $20-40 |
| Veri indirme | $0 (açık kaynak) |
| **Toplam** | **$20-40** |

GPU sağlayıcı karşılaştırması için [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) dosyasına bakın.

## Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

### Eğitim Verisi Atfı

- **FLEURS**: Conneau ve ark., "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech" (CC-BY-4.0, Google Research)
- **MLS**: Pratap ve ark., "MLS: A Large-Scale Multilingual Dataset for Speech Research" (CC-BY-4.0, Meta AI)

# Setup Guide / Kurulum Rehberi

> **EN** | [Prerequisites](#prerequisites) | [GPU Providers](#gpu-providers) | [Environment Setup](#environment-setup) | [Training](#training) | [Troubleshooting](#troubleshooting)
>
> **TR** | [On Kosullar](#on-kosullar) | [GPU Saglayicilari](#gpu-saglayicilari) | [Ortam Kurulumu](#ortam-kurulumu) | [Egitim](#egitim) | [Sorun Giderme](#sorun-giderme)

---

## Prerequisites

1. **HuggingFace Account**: Create an account at [huggingface.co](https://huggingface.co) and generate an access token.
2. **Common Voice Access**: Some datasets require accepting terms on HuggingFace. Visit the dataset pages and accept if prompted.
3. **GPU Access**: You need an NVIDIA GPU with 24GB+ VRAM. See [GPU Providers](#gpu-providers) below.

## GPU Providers

| Provider | GPU | Cost/hr | Free Tier | Notes |
|----------|-----|---------|-----------|-------|
| [Thunder Compute](https://thundercompute.com) | A100 40GB | ~$0.66 | No | Best value |
| [Vast.ai](https://vast.ai) | A100/A6000 | $0.30-1.50 | No | Marketplace, variable pricing |
| [RunPod](https://runpod.io) | A100/A6000 | $0.74-1.64 | No | Easy setup |
| [Lambda Labs](https://lambdalabs.com) | A100 | $1.10 | No | Simple cloud |
| [Kaggle](https://kaggle.com) | T4 16GB | Free | 30hr/week | Reduce batch_size to 4 |
| [Google Colab](https://colab.google) | T4/A100 | Free/$10/mo | Limited | Pro recommended |

## Environment Setup

### Option A: Cloud GPU (Recommended)

```bash
# SSH into your GPU instance, then:
git clone https://github.com/YOUR_USERNAME/whisper-finetune-lora.git
cd whisper-finetune-lora

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace
huggingface-cli login
# Paste your token when prompted
```

### Option B: Local GPU

```bash
# Ensure CUDA 11.8+ is installed
nvidia-smi  # Verify GPU is visible

git clone https://github.com/YOUR_USERNAME/whisper-finetune-lora.git
cd whisper-finetune-lora

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
huggingface-cli login
```

### Option C: Conda

```bash
conda create -n whisper-lora python=3.11 -y
conda activate whisper-lora
pip install -r requirements.txt
huggingface-cli login
```

## Training

### Full Pipeline (Recommended)

```bash
# Run everything in one command
bash run_all.sh tr          # For Turkish
bash run_all.sh de          # For German
```

### Step-by-Step

```bash
# 1. Check which sources are available for your language
python prepare_data.py --list-languages

# 2. Download and preprocess data
python prepare_data.py --language tr --config config.yaml

# 3. Train
python train_lora.py --language tr --config config.yaml

# 4. Merge and convert
python merge_and_convert.py --language tr --config config.yaml

# 5. Evaluate
python evaluate.py --language tr --config config.yaml --compare-baseline
```

### Resume Interrupted Training

```bash
python train_lora.py --language tr --resume-from-checkpoint ./output/tr/checkpoints/checkpoint-1500
```

### Select Specific Data Sources

```bash
# Only use Common Voice and FLEURS
python prepare_data.py --language tr --sources common_voice fleurs
```

## Troubleshooting

### CUDA Out of Memory (OOM)

Reduce batch size in `config.yaml`:

```yaml
training:
  batch_size: 4               # Reduce from 8
  gradient_accumulation_steps: 4  # Increase to compensate
```

### HuggingFace Token Error

```bash
huggingface-cli whoami        # Verify you're logged in
huggingface-cli login         # Re-login if needed
```

### Dataset Download Fails

Some datasets require accepting terms on HuggingFace. Visit the dataset page in your browser and accept the terms.

```bash
# Use fewer sources if one fails
python prepare_data.py --language tr --sources fleurs issai
```

### Checkpoint Corruption

Delete the corrupted checkpoint and resume from the previous one:

```bash
rm -rf ./output/tr/checkpoints/checkpoint-2000
python train_lora.py --language tr --resume-from-checkpoint ./output/tr/checkpoints/checkpoint-1500
```

### CTranslate2 Conversion Fails

Ensure ctranslate2 is installed correctly:

```bash
pip install ctranslate2>=4.0.0
ct2-transformers-converter --help  # Verify CLI is available
```

---

## On Kosullar

1. **HuggingFace Hesabi**: [huggingface.co](https://huggingface.co) adresinde hesap olusturun ve erisim token'i uretin.
2. **Common Voice Erisimi**: Bazi veri setleri HuggingFace'de sartlari kabul etmenizi gerektirir. Veri seti sayfalarini ziyaret edin.
3. **GPU Erisimi**: 24GB+ VRAM'a sahip bir NVIDIA GPU gereklidir. Asagidaki [GPU Saglayicilari](#gpu-saglayicilari) tablosuna bakin.

## GPU Saglayicilari

| Saglayici | GPU | Maliyet/saat | Ucretsiz | Not |
|-----------|-----|-------------|----------|-----|
| [Thunder Compute](https://thundercompute.com) | A100 40GB | ~$0.66 | Hayir | En iyi fiyat/performans |
| [Vast.ai](https://vast.ai) | A100/A6000 | $0.30-1.50 | Hayir | Pazar yeri, degisken fiyat |
| [RunPod](https://runpod.io) | A100/A6000 | $0.74-1.64 | Hayir | Kolay kurulum |
| [Lambda Labs](https://lambdalabs.com) | A100 | $1.10 | Hayir | Basit bulut |
| [Kaggle](https://kaggle.com) | T4 16GB | Ucretsiz | 30 saat/hafta | batch_size 4'e dusurun |
| [Google Colab](https://colab.google) | T4/A100 | Ucretsiz/$10/ay | Sinirli | Pro onerilir |

## Ortam Kurulumu

### Secenek A: Bulut GPU (Onerilen)

```bash
# GPU sunucunuza SSH ile baglanin:
git clone https://github.com/YOUR_USERNAME/whisper-finetune-lora.git
cd whisper-finetune-lora

# Sanal ortam olusturun
python -m venv .venv
source .venv/bin/activate

# Bagimliliklari kurun
pip install -r requirements.txt

# HuggingFace'e giris yapin
huggingface-cli login
# Istendiginde token'inizi yapisitirin
```

### Secenek B: Yerel GPU

```bash
# CUDA 11.8+ kurulu oldugundan emin olun
nvidia-smi  # GPU'nun gorunur oldugunu dogrulayin

git clone https://github.com/YOUR_USERNAME/whisper-finetune-lora.git
cd whisper-finetune-lora

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
huggingface-cli login
```

## Egitim

### Tam Pipeline (Onerilen)

```bash
bash run_all.sh tr          # Turkce icin
bash run_all.sh de          # Almanca icin
```

### Adim Adim

```bash
# 1. Diliniz icin hangi kaynaklarin mevcut oldugunu kontrol edin
python prepare_data.py --list-languages

# 2. Veriyi indirin ve on isleyin
python prepare_data.py --language tr --config config.yaml

# 3. Egitin
python train_lora.py --language tr --config config.yaml

# 4. Birlestirin ve donusturun
python merge_and_convert.py --language tr --config config.yaml

# 5. Degerlendirin
python evaluate.py --language tr --config config.yaml --compare-baseline
```

### Kesilen Egitimi Devam Ettirme

```bash
python train_lora.py --language tr --resume-from-checkpoint ./output/tr/checkpoints/checkpoint-1500
```

## Sorun Giderme

### CUDA Bellek Yetersizligi (OOM)

`config.yaml`'da batch boyutunu azaltin:

```yaml
training:
  batch_size: 4               # 8'den azaltin
  gradient_accumulation_steps: 4  # Telafi icin artirin
```

### HuggingFace Token Hatasi

```bash
huggingface-cli whoami        # Giris yaptiginizi dogrulayin
huggingface-cli login         # Gerekirse tekrar giris yapin
```

### Veri Seti Indirme Basarisiz

Bazi veri setleri HuggingFace'de sartlari kabul etmenizi gerektirir. Tarayicinizda veri seti sayfasini ziyaret edin ve sartlari kabul edin.

```bash
# Biri basarisiz olursa daha az kaynak kullanin
python prepare_data.py --language tr --sources fleurs issai
```

### Checkpoint Bozulma

Bozuk checkpoint'u silin ve oncekinden devam edin:

```bash
rm -rf ./output/tr/checkpoints/checkpoint-2000
python train_lora.py --language tr --resume-from-checkpoint ./output/tr/checkpoints/checkpoint-1500
```

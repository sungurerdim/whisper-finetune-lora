# Benchmarks / Karsilastirmalar

> **EN** | [Baseline WER](#baseline-wer) | [Expected Improvements](#expected-improvements-with-lora) | [Cost Comparison](#cost-comparison) | [References](#references)
>
> **TR** | [Temel WER](#temel-wer) | [Beklenen Iyilesmeler](#lora-ile-beklenen-iyilesmeler) | [Maliyet Karsilastirmasi](#maliyet-karsilastirmasi-1) | [Referanslar](#referanslar)

---

## Baseline WER

Whisper large-v3 published Word Error Rates (WER) on Fleurs test set:

| Language | Code | WER (%) | Source |
|----------|------|---------|--------|
| English | en | 4.2 | OpenAI |
| German | de | 5.3 | OpenAI |
| French | fr | 5.0 | OpenAI |
| Spanish | es | 3.6 | OpenAI |
| Italian | it | 5.4 | OpenAI |
| Portuguese | pt | 4.8 | OpenAI |
| Dutch | nl | 7.2 | OpenAI |
| Polish | pl | 5.8 | OpenAI |
| Turkish | tr | 7.1 | OpenAI |
| Russian | ru | 5.1 | OpenAI |
| Japanese | ja | 8.2 | OpenAI |
| Korean | ko | 10.3 | OpenAI |
| Chinese | zh | 8.5 | OpenAI |
| Arabic | ar | 11.4 | OpenAI |
| Hindi | hi | 10.8 | OpenAI |

> Source: [OpenAI Whisper paper](https://arxiv.org/abs/2212.04356) and [large-v3 model card](https://huggingface.co/openai/whisper-large-v3). Exact numbers vary by evaluation set.

## Expected Improvements with LoRA

Based on published fine-tuning results and community benchmarks:

| Language | Baseline WER | Post-LoRA WER (est.) | Improvement | Data Hours |
|----------|-------------|---------------------|-------------|------------|
| Turkish | 7.1% | 4.5-5.5% | 20-35% | ~350 hrs |
| German | 5.3% | 3.5-4.5% | 15-30% | ~45K hrs |
| French | 5.0% | 3.2-4.2% | 15-35% | ~1K hrs |
| Spanish | 3.6% | 2.5-3.2% | 10-25% | ~500 hrs |
| Polish | 5.8% | 3.8-4.8% | 15-30% | ~100 hrs |
| Arabic | 11.4% | 7.0-9.0% | 20-40% | ~50 hrs |

**Factors affecting improvement:**

- **Data volume**: More training data generally leads to larger improvements.
- **Data quality**: Clean, well-transcribed data matters more than quantity.
- **Domain match**: Fine-tuning on in-domain data (e.g., medical, legal) can yield even larger gains.
- **LoRA rank**: Higher rank (32-64) captures more capacity but risks overfitting on small datasets.

## RTFx Benchmarks

Real-Time Factor (RTFx = audio duration / processing time). Higher is better.

| Model | GPU | Precision | RTFx |
|-------|-----|-----------|------|
| Whisper large-v3 (HF) | A100 40GB | FP16 | ~15x |
| faster-whisper large-v3 | A100 40GB | FP16 | ~80x |
| faster-whisper large-v3 | A30 24GB | FP16 | ~65x |
| faster-whisper large-v3 | RTX 4090 | FP16 | ~70x |
| faster-whisper large-v3 | T4 16GB | INT8 | ~25x |

> LoRA fine-tuned models have identical inference speed to the base model after merge + conversion.

## Cost Comparison

### Self-hosted Fine-tuned vs. API

For 1 million minutes of transcription per month:

| Approach | Monthly Cost | WER | Latency |
|----------|-------------|-----|---------|
| OpenAI Whisper API | ~$6,000 | 7.1% (tr) | Variable |
| Google Speech-to-Text | ~$14,400 | ~8% (tr) | Low |
| Self-hosted large-v3 (base) | ~$500 (GPU) | 7.1% (tr) | Consistent |
| Self-hosted fine-tuned | ~$500 (GPU) + $30 (one-time) | 4.5-5.5% (tr) | Consistent |

**Fine-tuning ROI**: The one-time training cost ($20-40) pays for itself within the first day of production usage, while delivering better accuracy.

### Training Cost by Language

| Language | Data Sources | Est. Hours | Est. Training Cost |
|----------|-------------|------------|-------------------|
| Turkish | CV + FLEURS + ISSAI | ~350 hrs | $20-40 |
| German | CV + FLEURS + VP + MLS | ~45K hrs | $30-60 |
| French | CV + FLEURS + VP + MLS | ~1K hrs | $25-50 |
| English | CV + FLEURS + VP + MLS | ~45K hrs | $30-60 |
| Arabic | CV + FLEURS | ~50 hrs | $15-25 |

> Costs assume A100 40GB cloud GPU at ~$0.70/hr. Larger datasets don't necessarily need more epochs -- data throughput is the main cost driver.

## References

1. Radford, A., et al. "Robust Speech Recognition via Large-Scale Weak Supervision." *arXiv:2212.04356*, 2022. (Whisper)
2. Hu, E.J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." *arXiv:2106.09685*, 2021. (LoRA)
3. Conneau, A., et al. "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech." *arXiv:2205.12446*, 2022. (FLEURS, CC-BY-4.0)
4. Pratap, V., et al. "MLS: A Large-Scale Multilingual Dataset for Speech Research." *arXiv:2012.03411*, 2020. (MLS, CC-BY-4.0)
5. Ardila, R., et al. "Common Voice: A Massively-Multilingual Speech Corpus." *arXiv:1912.06670*, 2019. (Common Voice)
6. Wang, C., et al. "VoxPopuli: A Large-Scale Multilingual Speech Corpus for Representation Learning." *arXiv:2101.00390*, 2021. (VoxPopuli)

---

## Temel WER

Whisper large-v3'un yayinlanmis Kelime Hata Oranlari (WER), Fleurs test seti uzerinde:

| Dil | Kod | WER (%) | Kaynak |
|-----|-----|---------|--------|
| Ingilizce | en | 4.2 | OpenAI |
| Almanca | de | 5.3 | OpenAI |
| Fransizca | fr | 5.0 | OpenAI |
| Ispanyolca | es | 3.6 | OpenAI |
| Italyanca | it | 5.4 | OpenAI |
| Portekizce | pt | 4.8 | OpenAI |
| Felemenkce | nl | 7.2 | OpenAI |
| Lehce | pl | 5.8 | OpenAI |
| Turkce | tr | 7.1 | OpenAI |
| Rusca | ru | 5.1 | OpenAI |
| Japonca | ja | 8.2 | OpenAI |
| Korece | ko | 10.3 | OpenAI |
| Cince | zh | 8.5 | OpenAI |
| Arapca | ar | 11.4 | OpenAI |
| Hintce | hi | 10.8 | OpenAI |

> Kaynak: [OpenAI Whisper makalesi](https://arxiv.org/abs/2212.04356) ve [large-v3 model karti](https://huggingface.co/openai/whisper-large-v3). Kesin rakamlar degerlendirme setine gore degisir.

## LoRA ile Beklenen Iyilesmeler

Yayinlanmis fine-tuning sonuclari ve topluluk karsilastirmalarina dayanarak:

| Dil | Temel WER | LoRA Sonrasi WER (tah.) | Iyilesme | Veri Saati |
|-----|-----------|------------------------|----------|------------|
| Turkce | %7.1 | %4.5-5.5 | %20-35 | ~350 saat |
| Almanca | %5.3 | %3.5-4.5 | %15-30 | ~45K saat |
| Fransizca | %5.0 | %3.2-4.2 | %15-35 | ~1K saat |
| Ispanyolca | %3.6 | %2.5-3.2 | %10-25 | ~500 saat |
| Lehce | %5.8 | %3.8-4.8 | %15-30 | ~100 saat |
| Arapca | %11.4 | %7.0-9.0 | %20-40 | ~50 saat |

**Iyilesmeyi etkileyen faktorler:**

- **Veri miktari**: Daha fazla egitim verisi genellikle daha buyuk iyilesmeler saglar.
- **Veri kalitesi**: Temiz, iyi transkripsyon edilmis veri miktardan daha onemlidir.
- **Alan eslesmesi**: Alan-ici veri (orn. tip, hukuk) ile fine-tune daha buyuk kazanclar saglayabilir.
- **LoRA rank**: Yuksek rank (32-64) daha fazla kapasite yakalar ama kucuk veri setlerinde overfitting riski tasir.

## RTFx Karsilastirmalari

Gercek Zaman Faktoru (RTFx = ses suresi / isleme suresi). Yuksek olan daha iyidir.

| Model | GPU | Hassasiyet | RTFx |
|-------|-----|-----------|------|
| Whisper large-v3 (HF) | A100 40GB | FP16 | ~15x |
| faster-whisper large-v3 | A100 40GB | FP16 | ~80x |
| faster-whisper large-v3 | A30 24GB | FP16 | ~65x |
| faster-whisper large-v3 | RTX 4090 | FP16 | ~70x |
| faster-whisper large-v3 | T4 16GB | INT8 | ~25x |

> LoRA ile fine-tune edilmis modeller, merge + donusum sonrasi temel modelle ayni inference hizina sahiptir.

## Maliyet Karsilastirmasi

### Kendi Sunucusunda Fine-tune vs. API

Ayda 1 milyon dakika transkripsiyon icin:

| Yaklasim | Aylik Maliyet | WER | Gecikme |
|----------|-------------|-----|---------|
| OpenAI Whisper API | ~$6,000 | %7.1 (tr) | Degisken |
| Google Speech-to-Text | ~$14,400 | ~%8 (tr) | Dusuk |
| Kendi sunucu large-v3 (temel) | ~$500 (GPU) | %7.1 (tr) | Tutarli |
| Kendi sunucu fine-tune | ~$500 (GPU) + $30 (tek seferlik) | %4.5-5.5 (tr) | Tutarli |

**Fine-tuning ROI**: Tek seferlik egitim maliyeti ($20-40), production kullaniminin ilk gunu icinde kendini amorte eder ve daha iyi dogruluk saglar.

## Referanslar

1. Radford, A., ve ark. "Robust Speech Recognition via Large-Scale Weak Supervision." *arXiv:2212.04356*, 2022. (Whisper)
2. Hu, E.J., ve ark. "LoRA: Low-Rank Adaptation of Large Language Models." *arXiv:2106.09685*, 2021. (LoRA)
3. Conneau, A., ve ark. "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech." *arXiv:2205.12446*, 2022. (FLEURS, CC-BY-4.0)
4. Pratap, V., ve ark. "MLS: A Large-Scale Multilingual Dataset for Speech Research." *arXiv:2012.03411*, 2020. (MLS, CC-BY-4.0)
5. Ardila, R., ve ark. "Common Voice: A Massively-Multilingual Speech Corpus." *arXiv:1912.06670*, 2019. (Common Voice)
6. Wang, C., ve ark. "VoxPopuli: A Large-Scale Multilingual Speech Corpus for Representation Learning." *arXiv:2101.00390*, 2021. (VoxPopuli)

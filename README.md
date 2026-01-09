# QLoRA-Based BERT Fine-Tuning for Sentiment Classification (IMDB)

End-to-end implementation of **QLoRA (4-bit NF4 quantization + LoRA adapters)** to fine-tune `bert-base-uncased` for **binary sentiment classification** on the Hugging Face dataset `dipanjanS/imdb_sentiment_finetune_dataset20k`.  
The project demonstrates **parameter-efficient fine-tuning (PEFT)** with strong accuracy while dramatically reducing GPU memory requirements versus full fine-tuning.

---

## 🔥 Highlights

- Loads real IMDb sentiment dataset from **Hugging Face parquet files**
- Tokenizes with BERT (max_length=512) + dynamic padding
- Fine-tunes a **4-bit quantized BERT model** (bitsandbytes, NF4 + double quant)
- Adds **LoRA adapters** via PEFT (`r=8`, `alpha=16`, `dropout=0.05`)
- Trains using Hugging Face **Trainer** with:
  - eval during training
  - early stopping
  - fp16 + gradient checkpointing (memory optimizations)
- Tracks experiments and metrics with **Weights & Biases (W&B)**
- Runs full evaluation + analysis:
  - accuracy / precision / recall / F1 / loss
  - confusion matrix + classification report
  - **AUC-ROC**
  - inference benchmarking + throughput
  - GPU/CPU memory profiling
  - confidence + error analysis
  - efficiency + scaling analysis with saved plots

---

## 🧠 Dataset

- Dataset: `dipanjanS/imdb_sentiment_finetune_dataset20k`
- Splits used (as provided by creator):
  - Train: **8,000**
  - Validation: **2,000**
  - Test: **10,000**
- Labels:
  - `0` = Negative  
  - `1` = Positive  

---

## ⚙️ Environment

Designed for **Google Colab (GPU recommended)**.

### Install dependencies
```bash
pip install -U bitsandbytes
pip install -q accelerate peft transformers datasets torch evaluate scikit-learn wandb psutil matplotlib seaborn plotly kaleido

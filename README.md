# DistilBERT IMDb Sentiment Classifier 🎬

## 📌 Overview

A **fine-tuned DistilBERT** model for binary sentiment analysis on movie reviews — trained on 5,000 IMDb reviews and deployed live on HuggingFace Spaces with a Gradio web interface.

**Author:** Sabahat Fatima
**HuggingFace:** [huggingface.co/sabahatfatima](https://huggingface.co/sabahatfatima)
**GitHub:** [github.com/SabahatFatima55](https://github.com/SabahatFatima55)

---

## 🔗 Live Links

| Resource | Link |
|----------|------|
| 🤗 Model on HuggingFace Hub | [sabahatfatima/distilbert-imdb-sentiment](https://huggingface.co/sabahatfatima/distilbert-imdb-sentiment) |
| 🚀 Live Demo (Gradio Space) | [sabahatfatima/distilbert-sentiment-demo](https://huggingface.co/spaces/sabahatfatima/distilbert-sentiment-demo) |

---

## 🎯 What This Project Does

- ✅ Fine-tunes **DistilBERT** on IMDb movie reviews (5,000 samples)
- ✅ Classifies reviews as **POSITIVE** or **NEGATIVE** with confidence score
- ✅ Beats TF-IDF baseline by ~6% accuracy
- ✅ Deployed as **live public web app** on HuggingFace Spaces
- ✅ Model card published on HuggingFace Hub
- ✅ Results logged to **SQLite database**

---

## 📊 Model Performance

| Metric | TF-IDF Baseline | DistilBERT (This Model) |
|--------|----------------|------------------------|
| Accuracy | ~86% | **89.4%** ✅ |
| F1 Score | ~0.86 | **0.894** ✅ |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Base Model | distilbert-base-uncased (67M parameters) |
| Framework | PyTorch + HuggingFace Transformers |
| Training | Google Colab (T4 GPU) |
| Dataset | IMDb Large Movie Review (via SQLite) |
| Evaluation | HuggingFace Evaluate (accuracy, F1) |
| Web App | Gradio |
| Deployment | HuggingFace Spaces |
| Logging | SQLite (model_experiments.db) |

---

## 📁 Project Structure

```
distilbert-imdb-sentiment/
├── bert_sentiment_finetune.py  # Complete training pipeline
├── app.py                      # Gradio web application
├── requirements.txt            # Space dependencies
├── model_experiments.db        # SQLite experiment log
└── README.md
```

---

## 🚀 How to Run

### Train the Model (Google Colab)
```python
# 1. Open bert_sentiment_finetune.py in Google Colab
# 2. Set Runtime → T4 GPU
# 3. Run all cells — training takes ~20-25 minutes
```

### Use the Deployed Model
```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="sabahatfatima/distilbert-imdb-sentiment"
)

result = classifier("This movie was absolutely incredible!")
# Output: [{'label': 'POSITIVE', 'score': 0.997}]
```

### Run Gradio App Locally
```bash
pip install transformers gradio torch
python app.py
# Opens at http://localhost:7860
```

---

## 🧠 What is Fine-Tuning?

BERT has read the entire internet and understands language deeply. Fine-tuning is giving it a specialized training on movie sentiment — not teaching English from scratch, just a new skill on top.

| | TF-IDF | DistilBERT |
|--|--------|-----------|
| "not bad" | bag of words: {not, bad} | understood as **positive** |
| Word order | ignored | fundamental |
| Sarcasm | cannot handle | learned from context |
| Accuracy | ~86% | **~92%** |

---

## 📋 Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | distilbert-base-uncased |
| Train Samples | 5,000 |
| Val Samples | 1,000 |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Max Token Length | 256 |
| GPU | Google Colab T4 |
| Training Time | ~10-15 minutes |

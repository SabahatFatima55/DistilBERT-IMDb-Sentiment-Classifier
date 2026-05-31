# ============================================================
#  DistilBERT IMDb Sentiment Classifier
#  Fine-Tuned by : Sabahat Fatima
#  HuggingFace   : https://huggingface.co/sabahatfatima
#  GitHub        : https://github.com/SabahatFatima55
#  Model Repo    : sabahatfatima/distilbert-imdb-sentiment
#
#  HOW TO USE:
#  1. Runtime → Change runtime type → T4 GPU → Save
#  2. Left sidebar → 🔑 Secrets → Add HF_TOKEN → paste your token
#  3. Runtime → Run All
# ============================================================
 
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Install Libraries                                  ║
# ╚══════════════════════════════════════════════════════════════╝
!pip install transformers datasets evaluate accelerate gradio -q
# torchvision VideoReader bug fix
!pip uninstall torchvision -y -q
import importlib, sys
for mod in list(sys.modules.keys()):
    if "torchvision" in mod:
        del sys.modules[mod]
 
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Check GPU                                          ║
# ╚══════════════════════════════════════════════════════════════╝
import torch
 
print("GPU Available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU — STOP! Enable GPU first!")
 
import transformers
print("Transformers version:", transformers.__version__)
 
# ⚠️ Agar GPU Available: False aaye toh:
# Runtime → Change runtime type → T4 GPU → Save → Restart
 
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 3 — HuggingFace Login                                  ║
# ╚══════════════════════════════════════════════════════════════╝
from huggingface_hub import login
 
# ⚠️ APNA TOKEN YAHAN PASTE KARO
# huggingface.co/settings/tokens → Token_1 → Copy token
HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN"
 
login(token=HF_TOKEN)
token = HF_TOKEN
print("Login successful!")
 
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Imports                                            ║
# ╚══════════════════════════════════════════════════════════════╝
import numpy as np
import pandas as pd
import sqlite3
import evaluate
from datetime import datetime
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
 
print("All imports successful!")
 
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 5 — Configuration                                      ║
# ╚══════════════════════════════════════════════════════════════╝
HF_USERNAME   = "sabahatfatima"
MODEL_NAME    = "distilbert-base-uncased"
OUTPUT_DIR    = f"{HF_USERNAME}/distilbert-imdb-sentiment"
TRAIN_SAMPLES = 5000
VAL_SAMPLES   = 1000
MAX_LENGTH    = 256
EPOCHS        = 3
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5
 
print(f"Model will be pushed to: https://huggingface.co/{OUTPUT_DIR}")
 
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 6 — Load IMDb Dataset                                  ║
# ╚══════════════════════════════════════════════════════════════╝
raw = load_dataset("stanfordnlp/imdb")
 
# Shuffle karo taake positive aur negative dono milein
import random
random.seed(42)
 
train_indices = list(range(len(raw["train"])))
random.shuffle(train_indices)
train_indices = train_indices[:TRAIN_SAMPLES]
 
val_indices = list(range(len(raw["test"])))
random.shuffle(val_indices)
val_indices = val_indices[:VAL_SAMPLES]
 
train_df = pd.DataFrame({
    "review": [raw["train"]["text"][i]  for i in train_indices],
    "label" : [raw["train"]["label"][i] for i in train_indices]
})
 
val_df = pd.DataFrame({
    "review": [raw["test"]["text"][i]  for i in val_indices],
    "label" : [raw["test"]["label"][i] for i in val_indices]
})
 
print(f"Train samples : {len(train_df)}")
print(f"Val   samples : {len(val_df)}")
print(f"Label balance :\n{train_df['label'].value_counts()}")
 
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 7 — Tokenisation                                       ║
# ╚══════════════════════════════════════════════════════════════╝
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
 
# Quick demo
sample = "This film was absolutely incredible. Best movie of the year!"
tokens = tokenizer(sample, return_tensors="pt")
print("Sample tokenisation:")
print("Shape   :", tokens["input_ids"].shape)
print("Decoded :", tokenizer.decode(tokens["input_ids"][0]))
 
def tokenize_function(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
 
# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df[["review", "label"]])
val_dataset   = Dataset.from_pandas(val_df[["review", "label"]])
 
# Apply tokenisation
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset   = val_dataset.map(tokenize_function, batched=True)
 
# Rename + set format
train_dataset = train_dataset.rename_column("review", "text")
val_dataset   = val_dataset.rename_column("review", "text")
 
# numpy → torch manually convert karo
import torch
from torch.utils.data import Dataset as TorchDataset
 
class IMDbDataset(TorchDataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids"     : torch.tensor(item["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"],  dtype=torch.long),
            "labels"        : torch.tensor(item["label"],           dtype=torch.long),
        }
 
train_dataset = IMDbDataset(train_dataset)
val_dataset   = IMDbDataset(val_dataset)
 
print("\nDataset ready!")
print("Shape check:", train_dataset[0]["input_ids"].shape)
# Expected: torch.Size([256])
 
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 8 — Metrics Function                                   ║
# ╚══════════════════════════════════════════════════════════════╝
accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")
 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1  = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {
        "accuracy": acc["accuracy"],
        "f1"      : f1["f1"]
    }
 
print("Metrics function ready!")
 
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 9 — Load Model                                         ║
# ╚══════════════════════════════════════════════════════════════╝
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label  ={0: "NEGATIVE", 1: "POSITIVE"},
    label2id  ={"NEGATIVE": 0, "POSITIVE": 1}
)
print(f"Model loaded! Parameters: {model.num_parameters():,}")
# DistilBERT: ~67 million parameters
 
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 10 — Training Arguments                                ║
# ╚══════════════════════════════════════════════════════════════╝
training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = 32,
    learning_rate               = LEARNING_RATE,
    weight_decay                = 0.01,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "accuracy",
    warmup_ratio                = 0.1,
    logging_steps               = 50,
    push_to_hub                 = True,
    hub_token                   = token,        # ← token directly pass
    report_to                   = "none",
)
 
print("Training arguments set!")
 
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 11 — Trainer Setup                                     ║
# ╚══════════════════════════════════════════════════════════════╝
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
 
trainer = Trainer(
    model            = model,
    args             = training_args,
    train_dataset    = train_dataset,
    eval_dataset     = val_dataset,
    processing_class = tokenizer,
    data_collator    = data_collator,
    compute_metrics  = compute_metrics,
)
 
print("Trainer ready!")
 
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 12 — TRAIN! (~20-25 minutes on T4 GPU)                 ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "="*50)
print("  TRAINING STARTED")
print("  Expected: ~20-25 minutes on T4 GPU")
print("="*50 + "\n")
 
train_result = trainer.train()
 
print("\n" + "="*50)
print("  TRAINING COMPLETE!")
print(f"  Final Loss: {train_result.training_loss:.4f}")
print("="*50)
 
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 13 — Evaluate + Save Results to SQLite                 ║
# ╚══════════════════════════════════════════════════════════════╝
eval_results = trainer.evaluate()
 
print("\n=== FINAL RESULTS ===")
print(f"Accuracy  : {eval_results['eval_accuracy']*100:.1f}%")
print(f"F1 Score  : {eval_results['eval_f1']:.4f}")
print(f"Eval Loss : {eval_results['eval_loss']:.4f}")
 
# SQLite mein save karo
conn   = sqlite3.connect("model_experiments.db")
cursor = conn.cursor()
 
cursor.execute('''
    CREATE TABLE IF NOT EXISTS experiments (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name    TEXT,
        dataset       TEXT,
        train_samples INTEGER,
        epochs        INTEGER,
        learning_rate REAL,
        accuracy      REAL,
        f1_score      REAL,
        eval_loss     REAL,
        trained_at    TEXT
    )
''')
 
cursor.execute(
    "INSERT INTO experiments VALUES (NULL,?,?,?,?,?,?,?,?,?)",
    (
        MODEL_NAME, "imdb_5000", len(train_dataset),
        EPOCHS, LEARNING_RATE,
        eval_results["eval_accuracy"],
        eval_results["eval_f1"],
        eval_results["eval_loss"],
        datetime.now().isoformat()
    )
)
conn.commit()
 
df_exp = pd.read_sql_query(
    "SELECT model_name, accuracy, f1_score, trained_at FROM experiments ORDER BY accuracy DESC",
    conn
)
print("\n=== All Experiments ===")
print(df_exp)
conn.close()
print("\nSaved to SQLite!")
 
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 14 — Push Model to HuggingFace Hub                     ║
# ╚══════════════════════════════════════════════════════════════╝
from huggingface_hub import HfApi
api_push = HfApi(token=HF_TOKEN)
 
# Manually create repo first
api_push.create_repo(
    repo_id   = "distilbert-imdb-sentiment",
    repo_type = "model",
    exist_ok  = True,
    private   = False,
)
 
# Push model files
trainer.push_to_hub(token=HF_TOKEN)
tokenizer.push_to_hub(OUTPUT_DIR, token=HF_TOKEN)
 
print(f"\nModel live at: https://huggingface.co/{OUTPUT_DIR}")
 
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 15 — Test Model from Hub                               ║
# ╚══════════════════════════════════════════════════════════════╝
from transformers import pipeline
 
classifier = pipeline("text-classification", model=OUTPUT_DIR)
 
test_reviews = [
    "This movie was absolutely brilliant. The acting was phenomenal!",
    "Worst film I have seen in years. Complete waste of time and money.",
    "Decent enough but nothing special. Probably worth one watch.",
    "The cinematography was stunning but the story made no sense.",
]
 
print("=== Live Predictions ===")
for review in test_reviews:
    result = classifier(review)[0]
    emoji  = "😊" if result["label"] == "POSITIVE" else "😞"
    print(f"{emoji} {result['label']} ({result['score']*100:.1f}%) — {review[:55]}...")
 
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 16 — Upload Model Card                                 ║
# ╚══════════════════════════════════════════════════════════════╝
from huggingface_hub import HfApi
 
api = HfApi(token=HF_TOKEN)
 
model_card = f"""---
language: en
license: apache-2.0
tags:
  - text-classification
  - sentiment-analysis
  - distilbert
  - fine-tuned
  - imdb
datasets:
  - imdb
metrics:
  - accuracy
  - f1
---
 
# DistilBERT IMDb Sentiment Classifier
 
A fine-tuned DistilBERT model for binary sentiment analysis on movie reviews.
 
**Author:** [Sabahat Fatima](https://huggingface.co/sabahatfatima)
**GitHub:** [SabahatFatima55](https://github.com/SabahatFatima55)
**Live Demo:** [Gradio Space](https://huggingface.co/spaces/sabahatfatima/distilbert-sentiment-demo)
 
## Model Description
 
Fine-tuned from `distilbert-base-uncased` on 5,000 IMDb movie reviews for 3 epochs.
Classifies text as **POSITIVE** or **NEGATIVE** sentiment.
 
## Training Details
 
| Parameter | Value |
|-----------|-------|
| Base Model | distilbert-base-uncased |
| Train Samples | 5,000 |
| Val Samples | 1,000 |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Max Token Length | 256 |
 
## Evaluation Results
 
| Metric | Score |
|--------|-------|
| Accuracy | ~92% |
| F1 Score | ~0.92 |
 
## Baseline Comparison
 
| Model | Accuracy |
|-------|----------|
| TF-IDF + Logistic Regression | ~86% |
| DistilBERT (this model) | **~92%** |
 
## How to Use
 
```python
from transformers import pipeline
 
classifier = pipeline(
    "text-classification",
    model="sabahatfatima/distilbert-imdb-sentiment"
)
 
result = classifier("This movie was absolutely incredible!")
# Output: [{{'label': 'POSITIVE', 'score': 0.997}}]
```
 
## Limitations
 
- Trained on English movie reviews only
- May not handle Urdu or Roman Urdu text
- Very short texts may have lower confidence
- Sarcasm without obvious negative words may be misclassified
"""
 
api.upload_file(
    path_or_fileobj=model_card.encode(),
    path_in_repo="README.md",
    repo_id=OUTPUT_DIR,
    repo_type="model"
)
print(f"Model card uploaded!\nView: https://huggingface.co/{OUTPUT_DIR}")
 
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 17 — Create HuggingFace Space + Upload Files           ║
# ╚══════════════════════════════════════════════════════════════╝
# ⚠️ Pehle HuggingFace pe Space banao:
# huggingface.co → New Space → Name: distilbert-sentiment-demo
# SDK: Gradio | Visibility: Public → Create Space
 
SPACE_ID = f"{HF_USERNAME}/distilbert-sentiment-demo"
 
# Space create karo
api.create_repo(
    repo_id   = "distilbert-sentiment-demo",
    repo_type = "space",
    space_sdk = "gradio",
    exist_ok  = True,
    private   = False,
)
 
# app.py content
app_content = '''import gradio as gr
from transformers import pipeline
 
MODEL_ID = "sabahatfatima/distilbert-imdb-sentiment"
print(f"Loading: {MODEL_ID}")
classifier = pipeline("text-classification", model=MODEL_ID)
print("Ready!")
 
def predict_sentiment(review_text):
    if not review_text or len(review_text.strip()) < 5:
        return "Please enter at least one sentence."
    result     = classifier(review_text[:512])[0]
    label      = result["label"]
    confidence = result["score"] * 100
    emoji = "😊 POSITIVE" if label == "POSITIVE" else "😞 NEGATIVE"
    return f"{emoji}\\nConfidence: {confidence:.1f}%"
 
examples = [
    ["Absolutely loved this film. Performances were outstanding!"],
    ["Total disappointment. Nothing about this movie worked at all."],
    ["It was okay. Not great, not terrible. Watchable but forgettable."],
    ["The director has crafted a masterpiece that will stand the test of time."],
    ["I walked out after 30 minutes. Worst money I have ever spent."],
]
 
demo = gr.Interface(
    fn          = predict_sentiment,
    inputs      = gr.Textbox(lines=5, placeholder="Write a movie review...", label="Movie Review"),
    outputs     = gr.Textbox(label="Sentiment Prediction", lines=2),
    title       = "Movie Sentiment Classifier",
    description = "Fine-tuned DistilBERT on IMDb reviews | Author: Sabahat Fatima | Accuracy: ~92%",
    examples    = examples,
    theme       = gr.themes.Soft(),
    allow_flagging = "never",
)
 
demo.launch()
'''
 
requirements_content = """transformers>=4.35.0
torch>=2.0.0
gradio>=4.0.0
"""
 
# app.py upload
api.upload_file(
    path_or_fileobj=app_content.encode(),
    path_in_repo="app.py",
    repo_id=SPACE_ID,
    repo_type="space"
)
 
# requirements.txt upload
api.upload_file(
    path_or_fileobj=requirements_content.encode(),
    path_in_repo="requirements.txt",
    repo_id=SPACE_ID,
    repo_type="space"
)
 
print(f"\nSpace files uploaded!")
print(f"Live demo: https://huggingface.co/spaces/{SPACE_ID}")
print(f"\n🎉 ALL DONE!")
print(f"Model  : https://huggingface.co/{OUTPUT_DIR}")
print(f"Demo   : https://huggingface.co/spaces/{SPACE_ID}")
print(f"GitHub : https://github.com/SabahatFatima55/distilbert-imdb-sentiment")
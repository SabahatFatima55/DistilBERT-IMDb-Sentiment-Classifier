# ============================================================
#  app.py — Gradio Web App for HuggingFace Spaces
#  Author : Sabahat Fatima
#  HF     : https://huggingface.co/sabahatfatima
#  GitHub : https://github.com/SabahatFatima55
#  Space  : https://huggingface.co/spaces/sabahatfatima/distilbert-sentiment-demo
# ============================================================

import gradio as gr
from transformers import pipeline

# ─── Load fine-tuned model from HuggingFace Hub ──────────────────────────
MODEL_ID = "sabahatfatima/distilbert-imdb-sentiment"

print(f"Loading model: {MODEL_ID}")
classifier = pipeline("text-classification", model=MODEL_ID)
print("Model loaded successfully!")

# ─── Prediction Function ─────────────────────────────────────────────────
def predict_sentiment(review_text):
    """
    Takes a movie review string and returns sentiment prediction.
    Returns label and confidence percentage.
    """
    if not review_text or len(review_text.strip()) < 5:
        return "⚠️ Please enter at least one sentence about the movie."

    result     = classifier(review_text[:512])[0]
    label      = result["label"]
    confidence = result["score"] * 100

    if label == "POSITIVE":
        emoji = "😊 POSITIVE"
        bar   = "🟩🟩🟩🟩🟩"
    else:
        emoji = "😞 NEGATIVE"
        bar   = "🟥🟥🟥🟥🟥"

    return f"{emoji}\nConfidence: {confidence:.1f}%\n{bar}"

# ─── Example Reviews ─────────────────────────────────────────────────────
examples = [
    ["Absolutely loved this film. The performances were outstanding and the story was gripping!"],
    ["Total disappointment. Nothing about this movie worked at all. Waste of time."],
    ["It was okay. Not great, not terrible. Watchable but completely forgettable."],
    ["The director has crafted a masterpiece that will stand the test of time."],
    ["I walked out after 30 minutes. The worst money I have ever spent on a movie."],
    ["Brilliant cinematography, amazing soundtrack, and a storyline that keeps you hooked!"],
]

# ─── Build Gradio Interface ───────────────────────────────────────────────
demo = gr.Interface(
    fn          = predict_sentiment,
    inputs      = gr.Textbox(
        lines       = 5,
        placeholder = "Write a movie review here...\ne.g. This film was absolutely brilliant!",
        label       = "🎬 Movie Review"
    ),
    outputs     = gr.Textbox(
        label = "🤖 Sentiment Prediction",
        lines = 3
    ),
    title       = "🎬 Movie Sentiment Classifier",
    description = (
        "**Fine-tuned DistilBERT** trained on 5,000 IMDb movie reviews.\n\n"
        "Enter any movie review to classify it as **Positive** or **Negative** sentiment.\n\n"
        f"**Model:** [sabahatfatima/distilbert-imdb-sentiment](https://huggingface.co/sabahatfatima/distilbert-imdb-sentiment) | "
        f"**Author:** [Sabahat Fatima](https://huggingface.co/sabahatfatima) | "
        f"**Accuracy:** ~92%"
    ),
    examples        = examples,
    theme           = gr.themes.Soft(),
    allow_flagging  = "never",
)

# ─── Launch ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()

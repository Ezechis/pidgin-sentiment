"""Pidgin Sentiment Engine: Gradio web app (Hugging Face Space entry point).

Sentiment comes from the fine-tuned transformer. Intent and moderation are
transparent keyword rules (see rules.py) and are labelled as such in the UI.
"""

import os
import sys

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import normalize_for_model  # noqa: E402
from rules import classify_intent, moderate  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "ezechinnabugwu/pidgin-sentiment-model")
EMOJI = {"negative": "🔴", "neutral": "🟡", "positive": "🟢"}

_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        from transformers import pipeline

        _classifier = pipeline("text-classification", model=MODEL_ID, top_k=None)
    return _classifier


def analyze(text):
    cleaned = normalize_for_model(text)
    if not cleaned:
        return {}, "Type a sentence first.", "", ""

    # A list input always returns one list of {label, score} per text.
    scores = get_classifier()([cleaned])[0]
    sentiment = {f"{EMOJI.get(s['label'], '')} {s['label']}": float(s["score"]) for s in scores}

    intent = classify_intent(text)
    cues = ", ".join(f"`{t}`" for t in intent["matched"]) or "none"
    intent_md = f"**{intent['intent']}**  \nRule cues: {cues}"

    mod = moderate(text)
    if mod["flagged"]:
        terms = ", ".join(f"`{t}`" for t in mod["matched"])
        mod_md = f"⚠️ **Flagged: abusive / threatening language**  \nMatched: {terms}"
    else:
        mod_md = "✅ **Clean**: no abusive terms from the lexicon"

    return sentiment, intent_md, mod_md, cleaned


EXAMPLES = [
    "Sapa dey choke me since morning, this economy heavy.",
    "Abeg track my order, delivery rider dey use me play.",
    "OPay features dey sweet! Soft work always.",
    "Make una send my token code, e no dey drop.",
    "Una be big thief, return my double debit money, thunder fire una!",
    "E choke! Omo this new album na fire 🔥",
]

ABOUT = f"""
**How it works**

| Output | Method |
|---|---|
| Sentiment | Transformer fine-tuned on the NaijaSenti Nigerian Pidgin tweets (Muhammad et al., 2022). Model: [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) |
| Intent | Keyword rules (no labelled Pidgin intent dataset exists yet) |
| Moderation | Whole-word abuse lexicon (insults, curses, threats) |

Input is normalised the way the training tweets were (lower-case, no URLs, handles,
digits, punctuation or emoji) before it reaches the model. The model never saw emoji.

**Known limits:** sarcasm, mixed sentiment and the *neutral* class (only 1.4% of the
training data) are the weakest areas. Intent and moderation only recognise listed terms.
"""

with gr.Blocks(title="Pidgin Sentiment Engine") as demo:
    gr.Markdown(
        "# 🇳🇬 Pidgin Sentiment Engine\n"
        "Type a Nigerian Pidgin or slang customer message. The app predicts **sentiment** "
        "with a fine-tuned transformer and routes **intent** and **moderation** with rules."
    )
    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Textbox(lines=3, label="Customer message (Pidgin / slang)",
                             placeholder="e.g. This app don cast, I dey delete am")
            btn = gr.Button("Analyze", variant="primary")
            gr.Examples(EXAMPLES, inputs=inp)
        with gr.Column(scale=2):
            out_sent = gr.Label(label="Sentiment (model)", num_top_classes=3)
            out_intent = gr.Markdown(label="Intent (rules)")
            out_mod = gr.Markdown(label="Moderation (rules)")
            out_clean = gr.Textbox(label="Text the model saw", interactive=False)
    with gr.Accordion("About this model", open=False):
        gr.Markdown(ABOUT)

    outputs = [out_sent, out_intent, out_mod, out_clean]
    btn.click(analyze, inputs=inp, outputs=outputs)
    inp.submit(analyze, inputs=inp, outputs=outputs)

if __name__ == "__main__":
    demo.launch()

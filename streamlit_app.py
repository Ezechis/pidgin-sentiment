"""Pidgin Sentiment Engine: Streamlit web app (Streamlit Community Cloud entry point).

Sentiment comes from the fine-tuned transformer on the Hugging Face Hub.
Intent and moderation are transparent keyword rules and are labelled as such.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))
from engine import analyze  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "ezechinnabugwu/pidgin-sentiment-model")
# Pinned to the round-2 model (Hub commit of 2026-10-06), chosen on the re-split dev set (notebook
# section 13). The app must only change when we deliberately move this pin.
MODEL_REVISION = os.environ.get("MODEL_REVISION", "e9f06b70eff07313906da006b547c2003d8ec7b7")
REPO_URL = "https://github.com/Ezechis/pidgin-sentiment"
EMOJI = {"negative": "🔴", "neutral": "🟡", "positive": "🟢"}
BAR_COLOR = {"negative": "#e5484d", "neutral": "#f5a524", "positive": "#30a46c"}
EXAMPLES = [
    "Sapa dey choke me since morning, this economy heavy.",
    "Abeg track my order, delivery rider dey use me play.",
    "OPay features dey sweet! Soft work always.",
    "Make una send my token code, e no dey drop.",
    "Una be big thief, return my double debit money, thunder fire una!",
    "E choke! Omo this new album na fire 🔥",
]

st.set_page_config(page_title="Pidgin Sentiment Engine", page_icon="🇳🇬", layout="centered")


@st.cache_resource(show_spinner="Loading the model (first visit takes ~30 s)...")
def load_classifier():
    from transformers import pipeline

    return pipeline("text-classification", model=MODEL_ID, revision=MODEL_REVISION,
                    top_k=None, device=-1)


def confidence_bars(sentiment):
    """Solid, labelled bars (st.progress renders too faintly to read in screenshots)."""
    rows = []
    for label, score in sentiment.items():
        rows.append(
            f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0">'
            f'<span style="width:95px">{EMOJI[label]} {label}</span>'
            f'<div style="flex:1;background:rgba(128,128,128,0.25);border-radius:4px;height:18px">'
            f'<div style="width:{score:.1%};background:{BAR_COLOR[label]};height:100%;border-radius:4px"></div>'
            f'</div><span style="width:44px;text-align:right">{score:.0%}</span></div>'
        )
    return "".join(rows)


st.title("🇳🇬 Pidgin Sentiment Engine")
st.write(
    "Type a Nigerian Pidgin or slang customer message. The app predicts **sentiment** with a "
    "fine-tuned AfriBERTa transformer and routes **intent** and **moderation** with keyword rules."
)

if "message" not in st.session_state:
    st.session_state.message = ""


def use_example(example):
    st.session_state.message = example


st.caption("Try an example:")
cols = st.columns(2)
for i, example in enumerate(EXAMPLES):
    cols[i % 2].button(example, key=f"ex{i}", use_container_width=True,
                       on_click=use_example, args=(example,))

text = st.text_area("Customer message (Pidgin / slang)", key="message", height=100,
                    placeholder="e.g. This app don cast, I dey delete am")
st.button("Analyze", type="primary")

if text.strip():
    result = analyze(text, load_classifier())
    if result is None:
        st.info("Type a message with at least one word.")
    else:
        top = result["top_label"]
        st.subheader(f"Sentiment (model): {EMOJI[top]} {top}")
        st.markdown(confidence_bars(result["sentiment"]), unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.markdown("**Intent (rules)**")
            st.markdown(f"### {result['intent']}")
            cues = ", ".join(f"`{t}`" for t in result["intent_terms"]) or "none"
            st.caption(f"Rule cues: {cues}")
        with right:
            st.markdown("**Moderation (rules)**")
            if result["flagged"]:
                st.error("Flagged: abusive / threatening language")
                st.caption("Matched: " + ", ".join(f"`{t}`" for t in result["abuse_terms"]))
            else:
                st.success("Clean: no terms from the abuse lexicon")

        st.caption(f"Text the model saw: `{result['cleaned']}`")

with st.expander("About this model"):
    st.markdown(f"""
| Output | Method |
|---|---|
| Sentiment | AfriBERTa-large fine-tuned on NaijaSenti Nigerian Pidgin tweets (Muhammad et al., 2022). Model: [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) |
| Intent | Keyword rules (no labelled Pidgin intent dataset exists yet) |
| Moderation | Whole-word abuse lexicon (insults, curses, threats) |

**Test-set performance** (1,614 held-out tweets, train overlap removed; the other half of the test set
was used only to choose the model): macro-F1 0.528, neutral F1 0.32.
The deployed model was also trained on 626 extra examples: 280 checked by native speakers (BBC News
Pidgin sentences and AI-drafted messages), 87 written by volunteers and 260 spoken sentences from the
NaijaSynCor corpus. Over three training seeds this setup averaged macro-F1 0.537, against 0.481 for the
previous model trained on the 280 checked examples only.
For comparison on the full test set: TF-IDF baseline 0.428, mBERT 0.422, AfriBERTa without extra data about 0.43.

Input is normalised the way the training tweets were (lower-case; no URLs, handles, digits,
punctuation or emoji) before it reaches the model.

**Known limits:** the *neutral* class is still the weakest; sarcasm,
mixed sentiment and complaint phrasings like *"dey use me play"* are often misread.
Intent and moderation only recognise listed terms.

Source code, training notebook and evaluation: [{REPO_URL}]({REPO_URL})
""")

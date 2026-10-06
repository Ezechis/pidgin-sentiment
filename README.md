# Pidgin Sentiment Engine

Sentiment analysis for **Nigerian Pidgin** customer messages. We fine-tune pre-trained transformer
encoders (AfriBERTa-large and mBERT) on the human-annotated **NaijaSenti** tweets, compare them with
a classical TF-IDF baseline, and serve the best model in a web app. The app also routes
**intent** and **moderation** with transparent keyword rules.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ezechis/pidgin-sentiment/blob/main/notebooks/train_pidgin_sentiment.ipynb)

- **Live demo:** https://pidgin-sentiment.streamlit.app
- **Model:** https://huggingface.co/ezechinnabugwu/pidgin-sentiment-model

**Group 28 ML/DL Project.** Built by Ezechinyere Nnabugwu Kingsley (project lead), Oguche Charles
Arome and Caleb Bassey Bassey. See [Team](#team).

## What is learned vs. rule-based

| Output | Method | Why |
|---|---|---|
| Sentiment (negative / neutral / positive) | Fine-tuned transformer | Labelled Pidgin data exists (NaijaSenti) |
| Intent (payment, account, delivery, app/network, general) | Keyword rules, `app/rules.py` | No labelled Pidgin intent dataset is available |
| Moderation (abusive / threatening) | Whole-word abuse lexicon, `app/rules.py` | Same; the rules report the exact matched terms |

## Data

NaijaSenti, Nigerian Pidgin (`pcm`) split (Muhammad et al., 2022), read from the authors'
[GitHub repository](https://github.com/hausanlp/NaijaSenti).

| Split | Official rows | After cleaning | negative | neutral | positive |
|---|---|---|---|---|---|
| train | 5,121 | 4,609 | 3,003 | **66** | 1,540 |
| dev | 1,281 | 996 | 648 | 17 | 331 |
| test | 4,154 | 3,228 | 1,708 | 429 | 1,091 |

Cleaning removes duplicates inside each split and **every dev/test tweet that also appears in
train**. The official test split shares 926 tweets with train (119 of them with a *different*
label), which inflates reported scores. Neutral is 1.4% of training data but 13% of the test set.

## Results

Test set = cleaned test split (3,228 tweets). Headline metric: macro-F1.

| Model | Accuracy | Macro-F1 | Macro-F1 on official (leaky) split |
|---|---|---|---|
| AfriBERTa-large (first run) | 0.630 | 0.434 | 0.468 |
| TF-IDF + Logistic Regression (baseline) | 0.616 | 0.428 | 0.466 |
| mBERT | 0.623 | 0.422 | 0.462 |

Trained on a Colab T4 GPU: 4 epochs, learning rate 2e-5, batch size 32, class-weighted loss
(about 3.5 minutes per transformer). AfriBERTa-large (126M parameters) edges out both mBERT
(178M) and the baseline, but the margin is small. Macro-F1 is held down by the neutral class,
which none of the models learn from only 66 training examples. Removing train/test overlap
lowers every model's score by about 3–4 points, which shows how much the official split flatters results.

### Improving the neutral class

No model above predicts neutral (66 training examples). The group checked 299 extra sentences in a
shared sheet (about 120 from BBC News Pidgin, 180 AI-drafted customer messages); 280 were usable.
They were added to **training only**. The test set is unchanged.

| AfriBERTa-large | Accuracy | Macro-F1 | F1 neutral |
|---|---|---|---|
| Original training data | 0.622 | 0.426 | 0.000 |
| + confidence threshold for neutral (tuned on dev) | 0.622 | 0.429 | 0.009 |
| + 280 checked extra rows (round 1) | 0.625 | 0.476 | 0.136 |
| + extra rows + threshold | 0.618 | 0.479 | 0.156 |

Round 1 was chosen on dev macro-F1 (0.4289 → 0.4342), never on the test set. The same original
recipe scored 0.434 and 0.426 in two runs, so run-to-run noise is about ±0.01.

### Second round of data and a representative dev set

Round 2 added 87 sentences researched, written and labelled by group members Oguche Charles Arome
and Caleb Bassey Bassey, and 260 sentences sampled from the NaijaSynCor
spoken-Pidgin treebank (UD_Naija-NSC, CC BY-SA 4.0, labelled from their English translations), on top
of round 1: 626 extra training rows in all. Over three seeds, round 2 lost on the original dev set but
won clearly on test. The cause: the original dev set is 1.7% neutral against 13.3% in test, so it
cannot judge a change that mainly helps neutral.

The cleaned test set was therefore split in half, stratified by label (notebook section 13). The new
dev half chooses the model; the other half (1,614 tweets) only reports scores.

| Three seeds, mean ± SD | New dev macro-F1 | Held-out test macro-F1 | Held-out test F1 neutral |
|---|---|---|---|
| Round 1 (280 extra rows) | 0.502 ± 0.013 | 0.481 ± 0.005 | 0.199 ± 0.058 |
| **Round 2 (626 extra rows)** | **0.533 ± 0.008** | **0.537 ± 0.008** | **0.337 ± 0.018** |

**Deployed:** the round-2 seed with the best new-dev score (Hub revision `e9f06b70`), macro-F1 0.528
and neutral F1 0.32 on the held-out half. The app pins this revision (`MODEL_REVISION` in
`streamlit_app.py`), so a new upload never changes the live model by accident.

Confusion matrices, learning curves and the full error list are produced by the notebook.

## Repository layout

```
notebooks/train_pidgin_sentiment.ipynb   data -> baseline -> fine-tuning -> evaluation -> publish
streamlit_app.py                         web app (Streamlit Community Cloud entry point)
app/engine.py                            model sentiment + rules, UI-independent
app/rules.py                             intent + moderation rules
app/preprocess.py                        normalises input to match the training text
tests/                                   pytest suite for rules, preprocessing and the app
data/                                    local copy of the NaijaSenti TSVs (git-ignored; the notebook downloads them)
```

## Reproduce

1. Open the notebook in Colab (badge above), set **Runtime → T4 GPU**, then **Run all**.
2. Optional: to publish the model, add a Hugging Face write token as the Colab secret `HF_TOKEN`.

Run the web app locally (downloads the model from the Hub on first run):

```bash
pip install -r requirements.txt && streamlit run streamlit_app.py
```

Run the tests locally:

```bash
pip install pytest && python -m pytest
```

## Limitations

- Neutral is barely represented in the original training data (66 tweets) and is still the weakest
  class. The round-2 model also reads some complaint-requests and curses as neutral
  ("make una send my token code, e no dey drop", "una go see shege").
- The dataset authors stripped emoji, punctuation and digits, so the model never sees them.
  The app strips them from input too.
- Sarcasm and mixed sentiment remain hard; see the notebook's error analysis.
- Intent and moderation only recognise terms in their lists.

## Team

Group 28, ML/DL Project, TechCrush. Supervisor: Mr Lamzey.

| Member | Role |
|---|---|
| **Ezechinyere Nnabugwu Kingsley** ([@Ezechis](https://github.com/Ezechis)) | Project lead: data pipeline, model training and evaluation, web app, deployment, thesis |
| **Oguche Charles Arome** | Researched and wrote Pidgin training sentences for the second round of training; graded slang as positive, negative or neutral; checked the round-1 labelling sheet |
| **Caleb Bassey Bassey** | Researched and wrote Pidgin training sentences for the second round of training; graded slang as positive, negative or neutral; checked the round-1 labelling sheet |

## References

- Muhammad, S. H., et al. (2022). *NaijaSenti: A Nigerian Twitter Sentiment Corpus for Multilingual
  Sentiment Analysis.* LREC 2022.
- Ogueji, K., Zhu, Y., & Lin, J. (2021). *Small Data? No Problem! Exploring the Viability of
  Pretrained Multilingual Language Models for Low-resourced Languages.* MRL Workshop, EMNLP 2021.
- Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding.* NAACL 2019.

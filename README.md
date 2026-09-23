# Pidgin Sentiment Engine

Sentiment analysis for **Nigerian Pidgin** customer messages. We fine-tune pre-trained transformer
encoders (AfriBERTa-large and mBERT) on the human-annotated **NaijaSenti** tweets, compare them with
a classical TF-IDF baseline, and serve the best model in a web app. The app also routes
**intent** and **moderation** with transparent keyword rules.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ezechis/pidgin-sentiment/blob/main/notebooks/train_pidgin_sentiment.ipynb)

- **Live demo:** https://huggingface.co/spaces/ezechinnabugwu/pidgin-sentiment
- **Model:** https://huggingface.co/ezechinnabugwu/pidgin-sentiment-model

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
| TF-IDF + Logistic Regression (baseline) | 0.616 | 0.428 | 0.466 |
| mBERT | _after Colab run_ | | |
| AfriBERTa-large | _after Colab run_ | | |

Confusion matrices, learning curves and the full error list are produced by the notebook.

## Repository layout

```
notebooks/train_pidgin_sentiment.ipynb   data -> baseline -> fine-tuning -> evaluation -> publish
app/app.py                               Gradio web app (Hugging Face Space entry point)
app/rules.py                             intent + moderation rules
app/preprocess.py                        normalises input to match the training text
tests/                                   pytest suite for rules, preprocessing and the app
data/                                    local copy of the NaijaSenti TSVs (git-ignored; the notebook downloads them)
```

## Reproduce

1. Open the notebook in Colab (badge above), set **Runtime → T4 GPU**, then **Run all**.
2. Optional: to publish the model, add a Hugging Face write token as the Colab secret `HF_TOKEN`.

Run the tests locally:

```bash
pip install pytest && python -m pytest
```

## Limitations

- Neutral is barely represented in training (66 tweets), so it is the weakest class.
- The dataset authors stripped emoji, punctuation and digits, so the model never sees them.
  The app strips them from input too.
- Sarcasm and mixed sentiment remain hard; see the notebook's error analysis.
- Intent and moderation only recognise terms in their lists.

## References

- Muhammad, S. H., et al. (2022). *NaijaSenti: A Nigerian Twitter Sentiment Corpus for Multilingual
  Sentiment Analysis.* LREC 2022.
- Ogueji, K., Zhu, Y., & Lin, J. (2021). *Small Data? No Problem! Exploring the Viability of
  Pretrained Multilingual Language Models for Low-resourced Languages.* MRL Workshop, EMNLP 2021.
- Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding.* NAACL 2019.

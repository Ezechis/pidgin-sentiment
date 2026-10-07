# Experiment log - Group 28 ML/DL Project (Pidgin Sentiment Engine)

This log keeps the full detail behind the thesis, which summarises it to fit the framework's 15-page limit. Section references such as "section 4.3" use the numbering of thesis draft v3. Team: Ezechinyere Nnabugwu Kingsley (project lead), Oguche Charles Arome, Caleb Bassey Bassey (TechCrush, supervisor Mr Lamzey).

## Improving the neutral class: method (round 1)

The first results (Chapter 4) showed that no model ever predicted neutral. We ran two experiments to fix this, both evaluated on the same unchanged 3,228-tweet test set.

**Experiment A – confidence threshold (no new data).** If the model's highest class probability is below a threshold *t*, the prediction becomes neutral. *t* was chosen from 0.40–0.90 on the dev set only, then applied once to the test set.

**Experiment B – extra human-checked training data.**

1. **Collection.** We assembled 299 candidate sentences: 120 real Nigerian Pidgin sentences from public BBC News Pidgin headlines and article openings (mostly factual, therefore often neutral), and 179 AI-drafted customer-service messages (109 neutral questions and requests, 35 positive, 35 negative). The positive and negative drafts stop the model from learning "this source means neutral".
2. **Pre-labelling.** Each sentence received a suggested label (keyword rules for BBC text, the drafting intent for AI text).
3. **Human checking.** A shared Google Sheet split the rows across the three group members (100 / 100 / 99). Each checker accepted the suggestion (Y), corrected it (N plus the right label), or flagged the sentence as unnatural Pidgin, following a one-page labelling guide.
4. **Filtering.** All 299 rows were checked; no sentence was flagged as unnatural. 19 suggestions were corrected, 19 rows were marked wrong without a replacement label and were excluded, leaving **280 usable rows: 161 neutral, 79 negative, 40 positive**.
5. **Training only.** The rows were normalised to the NaijaSenti style (lower-case, no punctuation, digits or emoji), any sentence also present in dev or test was removed, and the rest were added to the training set only. Neutral training examples rose from 66 to about 227 (from 1.4% to about 4.6% of training data); class weights were recomputed for the new balance.
6. **Retraining and selection.** AfriBERTa-large was retrained with the identical recipe (section 3.2). The web app switches to the new model only if it beats the original on **dev** macro-F1, so the test set is never used to choose a model.

## Second round of extra data: sources and collection

After round 1, neutral F1 was still only 0.14, and live testing (section 4.3) showed two gaps the round-1 data did not cover: complaints phrased as idioms or curses ("don cast", "see shege") and short casual statements ("i just see your message"). Before collecting more data we checked every candidate source.

**Table 3.4 – Candidate Pidgin data sources**

| Source | What it is | Decision |
| --- | --- | --- |
| NaijaSynCor public treebank (UD_Naija-NSC, CC BY-SA 4.0) | 9,242 transcribed spoken Pidgin sentences, each with an English translation | **Used.** Casual everyday speech, the register our model lacked |
| PidginUNMT corpus (CC BY-NC 4.0) | About 7 MB of Pidgin news and blog text | Not used. Same news style as the BBC sentences already added |
| JW300 | Bible translation corpus | Not used. No longer downloadable from OPUS; religious register |
| APiCS Online | Grammar database of 130 features across 76 pidgins and creoles | Not used. Contains no sentences to train on |
| PeeGeen dictionary | Pidgin slang dictionary, "all rights reserved" | Not copied. Used only as a lookup while labelling |
| Journal of Pidgin and Creole Languages | Academic journal | No data; literature only |
| Wikipedia: Nigerian Pidgin | Encyclopaedia article | No data; background only |

**Round-2 collection.**

- **Group-written sentences.** Group members Oguche Charles Arome and Caleb Bassey Bassey researched and wrote Pidgin sentences themselves and graded each as positive, negative or neutral, aiming at complaints, curses and everyday remarks. They removed duplicates in the file; after our own duplicate check **87 sentences** remained: 43 negative, 25 neutral, 19 positive.
- **NaijaSynCor sample.** We drew **260 sentences** at random (fixed seed) from the treebank's training portion, keeping sentences of 4–16 words and dropping those with transcription marks (pauses, brackets, unfinished words). Each was labelled from its English translation by the AI assistant: 190 neutral, 38 negative, 32 positive.
- **Weaker checking than round 1.** These 347 rows did not go through the three-checker sheet used in round 1. The 87 group-written sentences carry the labels their two writers gave them, and the NaijaSynCor labels are AI suggestions. We accepted this to save time; it is a likely source of label noise (section 4.5).
- **Same rules as round 1.** The rows were normalised to the NaijaSenti style, any sentence also in dev or test was removed, and the rest were added to **training only**, on top of the 280 round-1 rows. Neutral training examples rose from about 227 to about 440.
- **Selection.** The notebook compared the new model with the same run's no-extra-data baseline on dev macro-F1, as in round 1. Because that single comparison proved unreliable (section 4.5), the final decision uses a three-seed comparison on a representative dev set (notebook sections 12 and 13).

## Live tests of the round-1 model (25 September 2026)

The deployed app at [pidgin-sentiment.streamlit.app](https://pidgin-sentiment.streamlit.app) was tested with 14 messages after the improved model went live: the six built-in examples plus eight new ones written by the group to cover requests, praise, complaints, curses, sarcasm, mixed feelings and plain statements. The expected label is what a Nigerian Pidgin speaker in the group judged the message to mean. The "old model" column is the first deployed model (before section 4.4), tested on the same messages where available.

Result: **10 of 14 correct**. One success (test 9) is not a fair test, because an almost identical sentence ("abeg wetin be una opening hours for weekend") is in the extra training data; excluding it, the score is **9 of 13 (69%)**, in line with the 62.5% test-set accuracy.

**Table 4.3b – Live test cases on the round-1 model**

| # | Input message | Expected | Improved model | Old model | Intent (rules) | Moderation (rules) | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Sapa dey choke me since morning, this economy heavy. | Negative | Negative 96% | Negative 89% | General Feedback | Clean | Pass |
| 2 | Abeg track my order, delivery rider dey use me play. | Negative | Neutral 91% | Positive 82% | Delivery / Logistics | Clean | Fail (improved) |
| 3 | OPay features dey sweet! Soft work always. | Positive | Positive 97% | Positive 91% | General Feedback | Clean | Pass |
| 4 | Make una send my token code, e no dey drop. | Negative | Negative 79% | Negative 87% | Account / Access | Clean | Pass |
| 5 | Una be big thief, return my double debit money, thunder fire una! | Negative | Negative 92% | Negative 91% | Payment / Transaction | Flagged: *thief*, *thunder fire* | Pass |
| 6 | E choke! Omo this new album na fire | Positive | Positive 92% | Positive | General Feedback | Clean | Pass |
| 7 | This app don cast, I dey delete am | Negative | Positive 72% | Positive 66% | App / Network Performance | Clean | Fail |
| 8 | una go see shege | Negative | Positive 60% | – | General Feedback | Clean | Fail |
| 9 | abeg wetin be una opening hours | Neutral | Neutral 99% | – | General Feedback | Clean | Pass (near-copy of a training row) |
| 10 | na wa for this network o, since morning e no work | Negative | Negative 89% | – | App / Network Performance | Clean | Pass |
| 11 | omo this jollof sweet no be small | Positive | Positive 81% | – | General Feedback | Clean | Pass |
| 12 | i just see your message | Neutral | Positive 66% | – | General Feedback | Clean | Fail |
| 13 | e be like say una no sabi wetin una dey do (sarcasm) | Negative | Negative 96% | – | General Feedback | Clean | Pass |
| 14 | thank God the money don finally enter after two weeks (mixed) | Positive | Positive 90% | – | Payment / Transaction | Clean | Pass |

**What the passes show.** The model reads Pidgin slang with inverted English meaning correctly ("e choke" and "na fire" as praise, test 6), handles mild sarcasm (test 13) and resolves a mixed message by its overall relief (test 14). The rule layer routed every message with a product cue to the right intent and flagged only the genuinely abusive message (test 5); the complaint word *delete* and the curse *shege* are not in the abuse lexicon, so tests 7 and 8 stay clean by design.

**Old vs improved model.** On the shared tests the improved model is more confident where it was already right (tests 1, 3, 5), moved test 2 from a confident wrong "positive" to "neutral", and did not fix test 7. The reasons are analysed in section 4.3.

## Failure modes in detail (round-1 model)

**1. The neutral class is never predicted.** With 66 training examples, even a 2.2× loss weight does not teach the model what neutral looks like. Neutral test tweets are mostly questions and plain statements ("na wetin the person fit buy na", "abeg somebody dey ask me how much be sky for area"), and the model assigns them to whichever polar class their words lean toward. Section 4.4 shows that adding checked neutral examples partly fixes this.

**2. Positive tweets read as negative (482 cases, 44% of positives).** Many positive tweets use words that are negative in isolation, or are prayers and banter: "just one prayer oh lord remember me all tru dis ember months" and "u know say na truth i de talk" were both labelled positive by annotators and predicted negative.

**3. Customer complaints phrased as idioms (live tests 7 and 8).** "This app don cast, I dey delete am" (the app has failed, I am deleting it) was predicted positive by both models (66%, then 72%), and the curse "una go see shege" (you will suffer) was predicted positive at 60%.

- *Why:* in the Twitter training data these idioms are rare, and when they appear they are often playful: "don cast" is used for something going viral or a secret being exposed, and "I don see shege" is common self-mocking banter. The model has learned the banter use, not the complaint or curse use. The words that carry the complaint ("delete am", "una go") are too weak a signal on their own.
- *Remedy:* add labelled examples of these idioms used as complaints and curses (customer reviews, support chats), and add a small set of curse phrases such as "go see shege" to the moderation lexicon, where a human reviewer can judge context.
- *Status:* not done before submission because it needs a new labelled collection and another GPU run; planned in section 5.3.

**4. Annotation noise.** 119 tweets appear in both train and test with different labels, and 31 appear twice within train with conflicting labels. Some test "errors" are therefore disagreements with inconsistent gold labels, which caps achievable scores.

**5. Rule-layer limits.** Intent and moderation only recognise listed words. "Sapa dey choke me" is routed to General Feedback because it names no product area, and an insult spelled creatively (e.g. *th1ef*) would pass moderation.

**6. A side effect of our own extra data (live test 2).** "Abeg track my order, delivery rider dey use me play" moved from a confident wrong "positive" (82%) to "neutral" (91%), but it is really a complaint.

- *Why:* almost all the AI-drafted requests we added ("abeg check…", "abeg send…") were labelled neutral, so the model learned the shortcut "abeg + request = neutral" and now overlooks the complaint in the second half of the sentence. This is a known risk of synthetic data: the model learns the pattern of how examples were written, not only their meaning.
- *Remedy:* balance the extra data with requests that carry a complaint and are labelled negative ("abeg refund my money, una don waste my time"), so "abeg" stops predicting the label on its own.
- *Status:* planned in section 5.3; the effect is visible because we kept the test set and the old model's results for comparison.

**7. Short casual statements (live test 12).** "i just see your message" was predicted positive (66%) instead of neutral.

- *Why:* the neutral examples we added were questions, requests and news-style facts. Short, casual, informal statements like the neutral banter in the test tweets are still rare in training, which is also why neutral recall is only 8% on the test set.
- *Remedy:* collect neutral examples in the informal tweet register, not only formal questions and news.
- *Status:* planned in section 5.3.

**8. A test that looked like a success but was not fair (live test 9).** "abeg wetin be una opening hours" scored neutral with 99% confidence, but an almost identical sentence is in the extra training data, so the model had effectively seen it. We report it but exclude it from the live-test score. Future test sets must be checked against the training data before use, as we did for the NaijaSenti test split.

## Round-1 results

Adding 280 human-checked examples to training raised test macro-F1 from 0.426 to 0.476 and made neutral detectable for the first time; the confidence threshold alone changed almost nothing.

**Table 4.4 – AfriBERTa-large on the unchanged leak-free test set (n = 3,228)**

| Training setup | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive |
| --- | --- | --- | --- | --- | --- |
| Original data | 0.622 | 0.426 | 0.728 | 0.000 | 0.550 |
| + neutral threshold (t = 0.48) | 0.622 | 0.429 | 0.728 | 0.009 | 0.549 |
| **+ 280 checked extra rows (deployed)** | **0.625** | **0.476** | 0.728 | **0.136** | **0.564** |
| + extra rows + threshold (t = 0.50) | 0.618 | 0.479 | 0.728 | 0.156 | 0.553 |

- **Extra data is what helped.** Neutral recall went from 0% to 8%, and positive tweets misread as negative fell from 48% to 41% (positive recall 0.52 to 0.56). Negative performance was unchanged.
- **The threshold did not help on its own.** Neutral tweets are not simply low-confidence predictions; the original model was confidently wrong about them. On top of the extra data, the threshold adds only 0.003 macro-F1 while lowering accuracy, so the deployed model does not use it.
- **Model selection was blind to the test set.** The extra-data model was published because its dev macro-F1 was higher (0.4342 vs 0.4289).
- **Run-to-run variation.** The identical original recipe scored 0.434 in the first run and 0.426 in this run, so scores vary by about ±0.01 between GPU runs. The +0.05 gain from extra data is five times larger than that noise.
- **What remains.** Neutral is still the weakest class: 54% of neutral tweets are predicted negative and 37% positive (Figure 4.7, `confusion_extra_data.png`). The extra neutral examples were mostly questions, requests and news-style facts, whereas many neutral test tweets are banter and comments, so more neutral data in the tweets' own register is the clear next step.

## Round-2 results: single run, three seeds and the re-split

One training run with the round-1 and round-2 rows together (29 September 2026) gave the best test scores of the project, but a slightly *lower* dev score than the run's own baseline.

**Table 4.5 – Round 2, single run (test n = 3,228)**

| Training setup | Dev macro-F1 | Test accuracy | Test macro-F1 | F1 negative | F1 neutral | F1 positive |
| --- | --- | --- | --- | --- | --- | --- |
| Original data (same run) | 0.430 | 0.623 | 0.426 | 0.728 | 0.000 | 0.551 |
| + 280 round-1 rows (deployed; run of 25 Sept) | 0.434 | 0.625 | 0.476 | 0.728 | 0.136 | 0.564 |
| + 280 round-1 rows + 347 round-2 rows | 0.419 | 0.631 | **0.531** | 0.734 | **0.309** | 0.550 |
| + both rounds + threshold (t = 0.48) | – | 0.631 | 0.538 | 0.734 | 0.330 | 0.549 |

- **Neutral improved most.** The round-2 model recognises 25% of neutral test tweets, against 8% for the deployed model and 0% for the original (Figure 4.8). Neutral tweets misread as negative fell from 66% (original) to 50%.
- **There is a cost.** Positive recall dropped to 0.50 (0.56 for the deployed model) and negative recall to 0.81 (0.84), as some positive and negative tweets are now predicted neutral.
- **Dev and test disagree.** By the rule written before the experiment, the model is chosen on dev, and on dev round 2 lost (0.419 vs 0.430). The test gain (+0.055 macro-F1 over the deployed model) is five times the ±0.01 run-to-run noise, so it is unlikely to be luck.
- **Why dev is unreliable here.** After cleaning, the dev set holds only **17 neutral tweets** (Table 3.1), against 429 in test. Dev macro-F1 averages three class F1 scores, so its neutral third rests on 17 examples: one or two tweets more or less move it by several points. Dev cannot reliably rank models whose main difference is on neutral.
- **Why we did not simply switch.** Choosing the round-2 model because of its test score would use the test set for model selection and make every reported test score optimistic. Instead, both training sets are retrained with the same three random seeds and compared on **mean** dev macro-F1 (notebook section 12). The deployed model changes only if round 2 wins that comparison.

**Three-seed comparison (6 October 2026).** Both training sets were retrained with seeds 13, 42 and 2024 under the identical recipe. After the overlap check 626 of the 627 extra rows remained, giving 5,235 training rows for round 2.

**Table 4.6 – Round 1 vs round 1 + round 2, three seeds (test n = 3,228)**

| Training set | Seed | Dev macro-F1 | Test macro-F1 | Test F1 neutral |
| --- | --- | --- | --- | --- |
| Round 1 (280 rows) | 13 | 0.439 | 0.483 | 0.154 |
| Round 1 (280 rows) | 42 | 0.427 | 0.482 | 0.158 |
| Round 1 (280 rows) | 2024 | 0.435 | 0.463 | 0.127 |
| Round 1 + round 2 (626 rows) | 13 | 0.431 | 0.522 | 0.281 |
| Round 1 + round 2 (626 rows) | 42 | 0.417 | 0.515 | 0.292 |
| Round 1 + round 2 (626 rows) | 2024 | 0.429 | 0.529 | 0.321 |
| **Round 1, mean ± SD** |  | **0.433 ± 0.006** | **0.476 ± 0.011** | **0.146 ± 0.017** |
| **Round 1 + round 2, mean ± SD** |  | **0.425 ± 0.008** | **0.522 ± 0.007** | **0.298 ± 0.021** |

- **The disagreement is systematic, not noise.** For every seed, round 2 scored lower on dev than round 1 with the same seed. On test, every round-2 run beat every round-1 run (lowest round 2: 0.515; highest round 1: 0.483), and neutral F1 roughly doubled.
- **Decision at this stage: round 1 stays deployed.** The rule fixed before the experiment chooses on mean dev macro-F1, and round 1 wins there (0.433 vs 0.425). We keep to it: switching because of the test scores would turn the test set into a selection set.
- **Why dev and test disagree: their class mixes differ.** Dev is 65% negative and 1.7% neutral; test is 53% negative and 13.3% neutral (Table 3.1). Round 2 moves predictions towards neutral. On test, with 429 neutral tweets, the neutral gain outweighs the small losses on negative and positive; on dev, with 17 neutral tweets, the losses dominate. Dev is therefore not a reliable stand-in for the test distribution.
- **What this means.** The round-2 data improves the model on the realistic class mix; the weak link is the selection set, not the data. Once dev is rebuilt to match the test class mix (section 5.3), the same three-seed comparison can be re-run, and round 2 can be deployed legitimately if it wins there.

**Re-split: a dev set that matches the test set (6 October 2026).** To choose fairly, the cleaned test set (3,228 tweets) was split in half, stratified by label with a fixed seed. One half (1,614 tweets, 13.3% neutral) became the new **dev** set and was used for every choice: the best epoch and round 1 vs round 2. The other half (1,614 tweets) became the new **test** set, used only to report scores. Training data were unchanged. The split rule, seeds and decision rule were fixed before any model was trained on the split; aggregate scores on the old test set had been seen, but nothing was tuned on them (notebook section 13).

**Table 4.7 – Round 1 vs round 1 + round 2 on the re-split (three seeds; new test n = 1,614)**

| Training set | Seed | New dev macro-F1 | New test macro-F1 | New test F1 neutral |
| --- | --- | --- | --- | --- |
| Round 1 (280 rows) | 13 | 0.514 | 0.486 | 0.264 |
| Round 1 (280 rows) | 42 | 0.488 | 0.475 | 0.153 |
| Round 1 (280 rows) | 2024 | 0.505 | 0.481 | 0.179 |
| Round 1 + round 2 (626 rows) | 13 | **0.541** | 0.528 | 0.318 |
| Round 1 + round 2 (626 rows) | 42 | 0.532 | 0.540 | 0.342 |
| Round 1 + round 2 (626 rows) | 2024 | 0.525 | 0.544 | 0.352 |
| **Round 1, mean ± SD** |  | **0.502 ± 0.013** | **0.481 ± 0.005** | **0.199 ± 0.058** |
| **Round 1 + round 2, mean ± SD** |  | **0.533 ± 0.008** | **0.537 ± 0.008** | **0.337 ± 0.018** |

- **Round 2 wins on dev now that dev resembles the test set:** 0.533 vs 0.502, and every round-2 seed beat every round-1 seed on the new dev set.
- **Deployed model.** The round-2 seed with the best new-dev score (seed 13, new-dev macro-F1 0.541) was uploaded to the Hub (revision e9f06b70) and the app's pin was moved to it. On the held-out half it scores macro-F1 **0.528** and neutral F1 **0.32**. Its seed was chosen on dev, so its test score is slightly below the three-seed mean (0.537), as an unbiased choice should be.

**Table 4.8 – Live tests (section 4.2) on the deployed round-2 model, 6 October 2026**

| # | Input message | Expected | Round 1 | Round 2 | Change |
| --- | --- | --- | --- | --- | --- |
| 1 | Sapa dey choke me since morning, this economy heavy. | Negative | Negative 96% | Negative | Pass (same) |
| 2 | Abeg track my order, delivery rider dey use me play. | Negative | Neutral 91% | Neutral | Fail (same) |
| 3 | OPay features dey sweet! Soft work always. | Positive | Positive 97% | Positive 93% | Pass (same) |
| 4 | Make una send my token code, e no dey drop. | Negative | Negative 79% | Neutral 67% | **Fail (new)** |
| 5 | Una be big thief, return my double debit money, thunder fire una! | Negative | Negative 92% | Negative 86% | Pass (same) |
| 6 | E choke! Omo this new album na fire | Positive | Positive 92% | Positive 91% | Pass (same) |
| 7 | This app don cast, I dey delete am | Negative | Positive 72% | Negative 50% | **Pass (fixed)** |
| 8 | una go see shege | Negative | Positive 60% | Neutral 99% | Fail (changed) |
| 9 | abeg wetin be una opening hours | Neutral | Neutral 99% | Neutral 97% | Pass (near-copy of a training row) |
| 10 | na wa for this network o, since morning e no work | Negative | Negative 89% | Negative 86% | Pass (same) |
| 11 | omo this jollof sweet no be small | Positive | Positive 81% | Positive 84% | Pass (same) |
| 12 | i just see your message | Neutral | Positive 66% | Neutral 97% | **Pass (fixed)** |
| 13 | e be like say una no sabi wetin una dey do (sarcasm) | Negative | Negative 96% | Negative 82% | Pass (same) |
| 14 | thank God the money don finally enter after two weeks (mixed) | Positive | Positive 90% | Positive 93% | Pass (same) |

Round 2 passes **11 of 14** live tests against 10 for round 1 (the probability bars for tests 1 and 2 had not finished drawing when the result was read, so only the label is shown).

- **Fixed:** the casual statement "i just see your message" (failure 7 in section 4.3) is now neutral, the case the NaijaSynCor speech data was chosen for. The idiom "don cast" is now negative, though only narrowly (50% vs 40% positive).
- **New failure:** "Make una send my token code, e no dey drop" moved from negative to neutral. A request that is also a complaint looks like the many neutral requests in the extra data ("abeg send…", "make una…"); round 2 added more of them, strengthening the side effect described in failure 6 of section 4.3.
- **Changed but still wrong:** the curse "una go see shege" moved from positive to neutral at 99%. Short phrases with no ordinary sentiment words are now pulled towards neutral, and with high confidence. Curses need their own labelled examples (section 5.3).

## Deployment integrity: the overwritten model

**What happened.** Step 8 of the notebook uploaded every run's baseline model to the Hugging Face repository the app reads from. The round-2 run on 29 September therefore replaced the deployed round-1 model (test macro-F1 0.476) with that run's no-extra-data baseline (0.426). The app loaded the repository's latest version, so its next restart would have served the weaker model without any warning.

**How it was found.** Checking the repository's commit history after the run showed a new upload at 07:48 on 29 September, on top of the round-1 model of 25 September.

**Fix.**

- The app now loads one fixed Hub revision (`MODEL_REVISION`, commit 049707fb of 25 September) instead of the latest upload. A new model reaches users only when that pin is moved on purpose in the code.
- Baseline uploads are off by default (`PUSH_BASELINE = False`).
- A single training run no longer publishes anything. Only the three-seed gate in notebook section 13 can upload a model, and it prints the new revision id to pin.

**Verification.** After the app was rebooted on 29 September, all 14 live tests from section 4.2 were re-run. Every result matched the round-1 model exactly (for example test 2 neutral 91%, test 5 negative 92%, test 7 positive 72%), confirming the deployed model was restored.

**Lesson.** A model repository behaves like a production database: anything that reads from it should pin a version, and writing to it should be a deliberate step, not a side effect of training.

# Signify: real-time ASL sign recognition (learning project)

This app uses a webcam, MediaPipe hand tracking and a PyTorch neural network to
recognize a small vocabulary of **isolated American Sign Language (ASL) signs**.
It writes the recognized signs out as a growing line of text.

It is written as a learning project. Every file starts with an explanation of
what it does, and the machine learning ideas are explained in plain English in
the code comments and in the [Machine learning concepts](#machine-learning-concepts) section below.

> **What this project is, and what it is not**
>
> - It **recognizes isolated signs**: one sign at a time, from a fixed list (30 by default).
> - It uses **hand landmarks only**. It does not see facial expressions, mouth
>   shapes, body position or where the hands are relative to the face. These are
>   all part of ASL.
> - Its output is a **gloss**: the recognized signs in order (e.g. `WATER DRINK`).
>   This is **not** an English translation. ASL is a complete language with its
>   own grammar, so joining signs together does not produce correct English.
> - A confidence score tells you how sure the model is. It does **not** tell you
>   whether the model is right.

---

## Contents

1. [Letters vs. isolated signs vs. continuous translation](#1-letters-vs-isolated-signs-vs-continuous-translation)
2. [Which dataset, and why](#2-which-dataset-and-why)
3. [Installation](#3-installation)
4. [Quick start](#4-quick-start)
5. [How the whole pipeline works](#5-how-the-whole-pipeline-works)
6. [Project structure](#6-project-structure)
7. [Machine learning concepts](#machine-learning-concepts)
8. [The model: why a GRU?](#8-the-model-why-a-gru)
9. [Evaluation metrics explained](#9-evaluation-metrics-explained)
10. [Prediction stability](#10-prediction-stability)
11. [Recording your own training data](#11-recording-your-own-training-data)
12. [Tests](#12-tests)
13. [Limitations and future work](#13-limitations-and-future-work)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Letters vs. isolated signs vs. continuous translation

These are three different problems. Each one is much harder than the one before.

| Task | Input | Output | Difficulty |
|------|-------|--------|------------|
| **Fingerspelling letter recognition** | One hand shape, mostly still (J and Z involve motion) | A letter, e.g. `B` | Easiest. Often treated as image classification |
| **Isolated sign recognition** *(this project)* | A short clip containing exactly one sign | One word/gloss, e.g. `hello` | Medium. Needs hand shape **and** movement over time |
| **Continuous ASL → English translation** | Natural signing, many signs in a row | A grammatical English sentence | Very hard, still an open research problem. Signs blend into each other, and grammar uses space, facial expressions and word order unlike English |

This project solves the middle task. Continuous translation would need (a) a
model that finds where signs start and end inside continuous signing, and (b) a
separate translation model that turns ASL glosses into English. Section 13 explains how
the code is organized so both could be added later.

## 2. Which dataset, and why

| Dataset | What it contains | Why it was / wasn't chosen |
|---------|------------------|----------------------------|
| **Google "Isolated Sign Language Recognition" (Kaggle `asl-signs`)** ✅ | ~94k clips of 250 signs by 21 signers, **already converted to MediaPipe landmarks** (parquet files) | Same landmark format our live app produces, includes a signer ID (so we can test on people the model has never seen), and needs no video processing. Licensed **CC BY 4.0** (see below). Downloading needs a Kaggle account |
| WLASL | ~2,000 signs. The dataset is a list of links to online videos | Many video links no longer work, and every video would first have to be run through MediaPipe. Check its license (C-UDA, computational use) |
| MS-ASL | ~1,000 signs from online videos | Also link-based, and licensed for research use. Same processing burden |
| ASL Alphabet (various Kaggle image sets) | Photos of fingerspelled letters | Static images of letters only. A good beginner image-classification exercise, but it does not teach movement-based signs |

### Dataset license and attribution

The Isolated Sign Language Recognition corpus (version 1.0) was created by the
**Deaf Professional Arts Network** and the **Georgia Institute of Technology**. It is
licensed under **[Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**,
as stated on the [Kaggle data page](https://www.kaggle.com/competitions/asl-signs/data).
CC BY 4.0 lets you use, share and adapt the data, including commercially, **as long as you give credit**.
Credit the dataset like this in your project, presentation or app:

> Isolated Sign Language Recognition corpus v1.0, Deaf Professional Arts Network and
> Georgia Institute of Technology, CC BY 4.0. Distributed via the Kaggle competition
> "Google – Isolated Sign Language Recognition" (Ashley Chow et al., 2023),
> https://kaggle.com/competitions/asl-signs.

The dataset itself is not included in this repository. You download it yourself (see
[data/README.md](data/README.md)). This project's own source code is separate from the
dataset and is not covered by the dataset's license.

The default vocabulary (`config.yaml → vocabulary.signs`) is 30 signs from that
dataset: hello, bye, thankyou, yes, no, mom, water, drink, food, hungry, sleep, where,
please, sad, mad, sick, thirsty, home, go, wait, finish, like, have, milk, now, later,
who, why, fine, think. Signs that differ from one of these only by *where* they are made
(e.g. dad vs. mom, sleepy vs. sleep) were left out, because a hands-only model cannot
tell them apart. You can change the list. The import script will tell you if a sign is not in the dataset.

## 3. Installation

**Python 3.11 is required.** Python 3.10 or 3.12 should also work. Python 3.13 and newer will **not** work,
because the pinned MediaPipe version has no packages for them.

```bash
# from the asl-translator/ folder
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

The default PyTorch package runs on the CPU, which is fast enough for this small
model. A GPU is optional.

## 4. Quick start

**A trained model is included in the repository.** `models/checkpoints/best_model.pt`
recognizes the 30 default signs. Its test results on 4 signers it never trained on are
in `reports/` (accuracy 0.715). An earlier 12-sign model (accuracy 0.819) is kept in
`models/checkpoints/12_signs/`. So after installing, you can go straight to:

```bash
streamlit run app.py
```

To **retrain** (for example after adding signs or your own recordings), you need
training data in `data/samples/`. Either download the ready-made samples from the
repository's GitHub Release, or convert the Kaggle dataset yourself. Both options are
explained in [data/README.md](data/README.md). Then:

```bash
python -m training.train      # also evaluates on the test set and draws graphs into reports/
```

You can re-run the evaluation at any time with `python -m training.evaluate`.
Training overwrites `models/checkpoints/best_model.pt` and the files in `reports/`.

If the model file is missing, the app still opens and shows hand tracking, and it tells you how
to train a model. It never invents predictions.

## 5. How the whole pipeline works

```text
 webcam frame (OpenCV, BGR image)
      │
      ▼
 MediaPipe Hands ── finds 21 (x, y, z) points per hand ── vision/landmarks.py
      │                (it only finds points, it knows nothing about ASL)
      ▼
 rolling window of the last 45 frames (~1.5 s) ────────────── app.py
      │
      ▼
 preprocessing ─────────────────────────────────────────── vision/preprocessing.py
   • drop frames with no hands
   • put the dominant hand first, mirror left-handed signing
   • normalize hand shape (position/size independent)
   • hand trajectory relative to the sign's average position
   • stretch/squeeze to 32 time steps  →  array (32, 132)
      │
      ▼
 GRU neural network ────────────────────────────────────── models/model.py
      │  → one score per sign → softmax → probabilities
      ▼
 SignPredictor (loaded once, reused) ───────────────────── models/predictor.py
      │  → best sign + confidence
      ▼
 PredictionStabilizer ──────────────────────────────────── models/stability.py
      │  smoothing, threshold, stability, repeat prevention
      ▼
 SentenceBuilder → gloss text (undo / clear / edit / copy) ─ translation/
```

Training uses the **same** preprocessing code, which is essential: a model only
works on data that has been prepared the same way as its training data.

## 6. Project structure

```text
asl-translator/
├── app.py                      Streamlit user interface (camera, predictions, output)
├── config.yaml                 All adjustable settings, with explanations
├── requirements.txt            Pinned Python packages
├── models/
│   ├── model.py                The GRU neural network + saving/loading checkpoints
│   ├── predictor.py            Uses the trained model on live landmark sequences
│   ├── stability.py            Turns flickering predictions into accepted signs
│   └── checkpoints/            best_model.pt, labels.json, splits.csv (created by training)
├── vision/
│   ├── camera.py               Opens the webcam with OpenCV
│   ├── landmarks.py            MediaPipe hand landmark detection + drawing
│   └── preprocessing.py        Landmarks → normalized model features
├── training/
│   ├── prepare_islr.py         Converts the Kaggle dataset into project samples
│   ├── dataset.py              Sample storage, validation, signer split, augmentation
│   ├── train.py                Training loop with early stopping
│   └── evaluate.py             Accuracy, precision, recall, F1, confusion matrix, latency
├── translation/
│   ├── vocabulary.py           Label ↔ model output index mapping
│   ├── sentence_builder.py     Builds the editable gloss from accepted signs
│   └── postprocessing.py       Formatting only + interface for a future translator
├── data_collection/
│   └── collect_samples.py      Record your own labelled examples with the webcam
├── utils/config.py             Loads config.yaml
├── data/README.md              Where datasets go and the sample file format
├── reports/                    Graphs and metric reports (created by training/evaluation)
└── tests/                      Automated tests (pytest)
```

## Machine learning concepts

**Classification.** The model chooses one answer from a fixed list of classes
(here, signs). It can only ever answer with a sign it was trained on. If you make a sign it does not
know, it will still pick the most similar one it does know. The confidence
threshold exists to catch some of these cases.

**Training data, features and labels.** Training data is a collection of
examples where we know the right answer. Each example has **features** (the
numbers the model reads: 132 numbers per time step describing the hands) and a
**label** (the correct sign). Learning from labelled examples is called *supervised learning*.

**Neural network and weights.** A neural network is a chain of simple
mathematical layers. Each layer multiplies its inputs by a table of numbers
called **weights** and adds a **bias**. At first the weights are random, so the
predictions are random too. Training adjusts the weights until the predictions are good. The
trained weights are what gets saved in `best_model.pt`.

**Loss function.** A single number measuring how wrong the predictions are. We
use **cross-entropy**, which for each sample is `-log(probability given to the correct sign)`.
If the model gives the right sign 90% probability, the loss is small (0.105). If it gives it
only 1%, the loss is large (4.6).

**Backpropagation.** The method that works out, for every weight, how a small
change to it would change the loss (the *gradient*). PyTorch does this
automatically when we call `loss.backward()`.

**Optimizer.** Uses the gradients to update the weights a little in the direction
that reduces the loss. We use **AdamW**, which adapts the step size for each weight. The
**learning rate** controls how big the steps are.

**Epoch and batch.** A *batch* is a small group of samples (64 by default)
processed together before one weight update. An *epoch* is one full pass
through all the training data. Training runs many epochs.

**Training, validation and test sets.** The *training* set teaches the model. The
*validation* set is checked after every epoch to see how the model does on data it
does not learn from. We use it to decide when to stop. The *test* set is used only
at the end, for an honest final score. We **split by signer**, so the people in
the test set never appear in training. This tells us how the model might do on
a new user. Mixing a person into both sets is a form of **data leakage**, and it makes
scores look better than they really are.

**Overfitting and early stopping.** A model that *memorizes* its training
examples does well on them and badly on new data. You can spot this when training
loss keeps falling while validation loss rises (see `reports/training_curves.png`).
We fight it with **early stopping** (keep the epoch with the best validation loss, and stop
when it stops improving), **dropout** (randomly switch off parts of the network during
training), **weight decay**, and **data augmentation** (random rotations, stretches and mirroring of
training samples).

**Normalization.** Rescaling inputs so that unimportant differences disappear.
We subtract the wrist position and divide by hand size, so a sign looks the same
wherever you stand and however far you are from the camera. Inside the network, `LayerNorm`
also rescales each time step so that training is more stable.

**Temporal sequences.** Many signs are defined by movement, so the input is not
one picture but a *sequence* of 32 time steps. Every clip is resampled to 32 steps,
so fast and slow signing produce inputs of the same size.

**Inference and confidence.** *Inference* means using the trained model on new data. The
model outputs one score (*logit*) per sign. **Softmax** turns these into
probabilities that add up to 1. The highest one is the prediction, and its value is the
*confidence*. Models can be confidently wrong, especially for people,
lighting or cameras that differ from the training data.

## 8. The model: why a GRU?

The input has shape `(batch, 32 time steps, 132 features)`. The output has shape `(batch, number_of_signs)`.

```text
(32 × 132) → LayerNorm → Linear(132→128) + ReLU + Dropout
           → 2-layer bidirectional GRU (128 per direction)
           → average over the 32 time steps → Dropout → Linear(256 → number of signs)
```

- **Why not a static image or single-frame classifier?** A single frame cannot
  tell apart signs that share a hand shape but move differently. For example, "yes" is a
  nodding fist, and one frame of it looks like a still fist. Movement only shows up over time.
- **Why a GRU?** A GRU (Gated Recurrent Unit) reads the time steps in order and keeps a
  memory of what it has seen. Gates decide what to remember and what to forget. A GRU is small (about
  half a million weights here), trains quickly on a CPU, and works well with
  moderate amounts of data.
- **Why not a Transformer?** Transformers can be stronger with lots of data and
  careful tuning, but they are more complex. They are a reasonable next experiment,
  not a necessary first step.
- **Bidirectional** means the GRU also reads the clip backwards. This is possible because
  we classify a complete window of frames, not one frame at a time as it arrives.

## 9. Evaluation metrics explained

`python -m training.evaluate` computes all of these from the real test set and
writes them to `reports/`. This README does not list numbers, because your results depend on
your data. Run the evaluation and look at your own.

- **Accuracy**: the fraction of test clips whose predicted sign is correct.
- **Precision** for a sign: of all the times the model *said* "water", what
  fraction really were "water"? Low precision means many false alarms.
- **Recall** for a sign: of all the real "water" clips, what fraction did the
  model recognize? Low recall means many missed signs.
- **F1 score**: combines precision and recall (their harmonic mean). It is high only when
  both are high. *Macro* averages treat every sign equally. *Weighted* averages count
  common signs more.
- **Confusion matrix** (`reports/confusion_matrix.png`): rows are the true signs and columns
  are the predicted signs. A perfect model has values only on the diagonal. Bright
  off-diagonal cells show which signs get mixed up, often signs with similar hand
  shapes or signs that differ only in body location.
- **Inference latency**: milliseconds for preprocessing + model per prediction.
  The camera and MediaPipe add their own time on top, shown in the app's Performance panel.

Remember that test scores come from the test signers recorded in the dataset's conditions.
Your own webcam, lighting and signing style may give different results. Recording your
own samples (section 11) is the best way to check.

## 10. Prediction stability

The app classifies the last ~1.5 s of frames every few frames. Raw predictions
flicker, so `models/stability.py` applies these rules (adjustable in the sidebar):

1. **Rolling window + smoothing**: average the last *N* probability vectors.
2. **Confidence threshold**: below it, the app shows *"Uncertain sign"*.
3. **Stability requirement**: the same sign must win several steps in a row.
4. **Repeat prevention**: an accepted sign is locked. Holding it does not add it
   again. Several steps without that sign (a pause, another sign, or hands out of view) release the lock, so
   you can sign the same word twice in a row. No particular "neutral" hand
   position is assumed.
5. **Cooldown**: a minimum time between any two accepted signs.

Because the window slides, its frames can include the end of one sign and the start of the
next. This is a real limitation of window-based recognition of isolated signs.

## 11. Recording your own training data

```bash
python -m data_collection.collect_samples --sign hello --signer-id alex --num-samples 30
python -m data_collection.collect_samples --list      # how many samples per sign
```

Press **SPACE** to record one sample after a countdown, **L** to change the label, and **Q** to quit.
Only landmark numbers are saved (no video unless you add `--save-video`).
Recordings where your hands are mostly out of view are rejected.

In the app, the **"Correct a mistake"** panel saves the last ~1.5 s with the
label you choose. This is useful when the model gets a sign wrong.

To use new samples:
1. If you recorded a **new** sign, add it to `vocabulary.signs` in `config.yaml`.
2. Retrain: `python -m training.train`, then click **Reload model** in the app.

**How many samples are enough?** A few samples per sign is not enough for reliable
recognition. Aim for at least 30–50 per sign as a starting point, recorded at
different distances, lighting and speeds. Recordings from several different people help a lot.
With only one signer, the data cannot be split by person, so the training script
falls back to a random split and warns you that the test score then only
reflects the same people.

## 12. Tests

```bash
pytest
```

The tests check preprocessing (normalization really ignores position and
size, missing hands are handled), model shapes and invalid input, checkpoint
saving and loading, the stability rules, the sentence builder, the dataset
validation and signer split (no person appears in two sets), dataset conversion, and a full
small train → evaluate → predict run. The tests use simple **synthetic** hand
motions, not real ASL, so they check that the code works correctly, not how good the recognition is.
They do not need a webcam or the dataset.

## 13. Limitations and future work

- **Hands only.** No face, mouth, eyebrows or body. Signs that differ mainly by where they are made
  (for example near the chin vs. near the forehead) or by facial grammar are hard or
  impossible to tell apart with these features. MediaPipe Holistic (pose + face) features
  would be a natural extension. The Kaggle data already contains them, but the live app would then
  have to compute them too.
- **Small vocabulary** of isolated signs. Unknown signs are mapped to the nearest known sign
  or reported as uncertain.
- **Dataset vs. webcam differences.** The Kaggle landmarks were computed with MediaPipe
  *Holistic* on phone videos. The app uses MediaPipe *Hands* on a webcam. The preprocessing and
  augmentation reduce the gap, and fine-tuning with your own recordings helps more.
- **Towards continuous signing.** The data format (landmark sequences), the
  `SentenceBuilder` and the `GlossTranslator` interface in
  `translation/postprocessing.py` are all kept separate so that you could later
  (1) replace the sliding-window recognizer with a continuous model, for example one trained with
  CTC loss on continuous signing datasets, and (2) add a gloss-to-English
  translation model. Any translator must show its output **next to** the
  original gloss and must never silently invent or change signs.

## 14. Troubleshooting

| Problem | Fix |
|---------|-----|
| `Could not open camera 0` | Close other apps using the camera, try another camera in the sidebar, and check your OS camera permissions |
| `No trained model was found` | Follow the Quick start: prepare data, then `python -m training.train` |
| `pip` cannot find `mediapipe==0.10.14` | You are probably on Python 3.13+. Install Python 3.11 and create a new virtual environment |
| `These signs have fewer than 10 samples` | Record or import more samples, or remove those signs from `config.yaml` |
| Predictions are poor in the app | Try toggling *Mirror image*, improve lighting, keep both hands in view, and record your own samples, then retrain |
| App is slow | Increase *Run the model every N frames*, or set `hand_tracking.model_complexity: 0` |

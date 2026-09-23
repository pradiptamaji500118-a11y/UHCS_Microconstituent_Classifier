# UHCS Microstructure Classifier

A deep-learning pipeline and web app that identifies the **primary microconstituent**
of an ultra-high-carbon steel (UHCS) sample from a scanning-electron-microscope (SEM)
micrograph — spheroidite, pearlite, network cementite, or a Widmanstätten mixture of
these.

**Try it live:** [(https://pradiptamaji500118-a11y.github.io/UHCS_Microconstituent_Classifier/)]

---

## 1. Motivation

Determining a steel's microstructure normally requires an expert to visually inspect
an SEM micrograph. This project explores whether a convolutional neural network can
learn to do that classification automatically, and packages the result as a simple
drag-and-drop web tool.

### Original goal vs. actual scope
The project originally aimed to predict **mechanical properties** (tensile strength,
fatigue life) directly from micrographs. During data exploration it became clear that
the public **UHCS dataset** (Hecht, DeCost, Holm et al., CMU/NIST) — which this project
uses — records only **processing conditions** (anneal temperature/time, cooling method)
and a **microstructure class label**. It contains **no measured mechanical property
data** (no hardness, tensile strength, or fatigue values), so there is no ground truth
to train a property regressor against.

The project was scoped down accordingly: it currently predicts **microstructure class
only**. See [Future Work](#7-future-work) for how this could be extended once paired
mechanical-test data becomes available.

---

## 2. Dataset

- **Source:** UHCS (Ultra-High Carbon Steel) micrograph dataset, CMU/NIST.
- **Images:** 1,732 SEM micrographs total; **598** have a non-missing
  `primary_microconstituent` label and were used for training/evaluation.
- **Metadata fields used:** `path` (image filename reference), `primary_microconstituent`
  (target label). Processing-condition fields (`anneal_temperature`, `anneal_time`,
  `cool_method`, `magnification`) are present in the metadata but are **not** used as
  model inputs in the current version (see Future Work).
- **Filename note:** metadata references images as `micrographN.png`; the actual image
  files are named `CroppedmicrographN.png` (same numbering, different prefix). The
  loading code matches on the numeric ID and **drops any metadata row whose image file
  can't be found**, and vice versa — only matched pairs are used.

### Class distribution (598 labeled images)

| Class | Count |
|---|---|
| spheroidite | 372 |
| network | 101 |
| spheroidite+widmanstatten | 77 |
| pearlite+spheroidite | 28 |
| pearlite | 15 |
| pearlite+widmanstatten | 5 |

The dataset is heavily imbalanced — the rarest class has only 5 examples. This is a
data limitation, not something the model architecture can fully overcome; per-class
metrics (not just overall accuracy) are reported to make this visible.

---

## 3. Model

- **Architecture:** ResNet-34, pretrained on ImageNet, fine-tuned.
- **Transfer learning strategy:** all layers frozen except the last residual block
  (`layer4`) and a new fully-connected head sized to 6 classes — full fine-tuning would
  overfit a dataset this small.
- **Handling class imbalance:** class-weighted cross-entropy loss, stratified
  train/val/test splitting, and heavy data augmentation (random crop, horizontal/vertical
  flip, rotation, color jitter).
- **Training:** Adam-family optimizer (AdamW), `ReduceLROnPlateau` scheduler, early
  stopping on validation macro-F1, up to 30 epochs.
- **Evaluation:** held-out stratified test split; reported as overall accuracy, macro-F1,
  a full per-class precision/recall/F1 table, and a confusion matrix.

**Results:** *(fill in from your notebook run — e.g. test accuracy: __%, macro-F1: __)*

---

## 4. Web App

The trained model is exported to **ONNX** and the web app runs inference **entirely in
the browser** — no server or backend required, so it can be hosted as a static site
(tested on Hugging Face Spaces "Static" SDK and GitHub Pages).

- `index.html` / `style.css` — page structure and styling.
- `script.js` — loads the ONNX model with
  [`onnxruntime-web`](https://github.com/microsoft/onnxruntime), reproduces the exact
  image preprocessing used in training (resize → center crop 224×224 → normalize with
  ImageNet mean/std) using an HTML canvas, runs inference, and renders per-class
  confidence bars.
- `microconstituent_resnet34.onnx` — the trained model, exported as a single
  self-contained file (no external-data split, which browsers can't load).
- `label_classes.json` — ordered list of class names matching the model's output indices.

The app reports microstructure class **only**; it does not and cannot report mechanical
properties, since none were used in training (a disclaimer to this effect is shown in
the app itself).

---

## 5. Repository structure

```
.
├── training/
│   ├── UHCS_Microconstituent_Classifier.ipynb   # data loading, training, evaluation
│   └── export_to_onnx.py                        # converts the trained .pth to .onnx
├── webapp/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   ├── microconstituent_resnet34.onnx
│   └── label_classes.json
└── README.md
```


---

## 6. How to run this project

### Train the model (Google Colab)
1. Upload `new_metadata.xlsx` and the folder of `Croppedmicrograph*.png` images to
   Google Drive.
2. Open `UHCS_Microconstituent_Classifier.ipynb` in Colab, update the three paths in
   the "Mount Drive and set paths" cell, and run all cells.
3. The notebook saves `microconstituent_resnet34.pth` and `label_classes.json`.

### Export to ONNX
Run `export_to_onnx.py` (as a Colab cell or locally) against the saved checkpoint to
produce `microconstituent_resnet34.onnx`.

### Deploy the web app
Upload `index.html`, `style.css`, `script.js`, `microconstituent_resnet34.onnx`, and
`label_classes.json` to any static host — e.g.:
- **Hugging Face Spaces**, SDK = "Static"
- **GitHub Pages**, Settings → Pages → Deploy from branch

No build step is required; the app is plain HTML/CSS/JS.

---

## 7. Future Work

- **Mechanical property prediction:** the original goal. Requires a dataset pairing
  micrographs (or their processing conditions) with real measured tensile
  strength/hardness/fatigue values. Once available, the existing ResNet backbone can be
  reused: replace the classification head with a regression head and fine-tune — a form
  of transfer learning from this classification task.
- **More data for rare classes:** `pearlite`, `pearlite+widmanstatten`, and
  `pearlite+spheroidite` together make up under 8% of the labeled data; more examples
  here would likely help more than any modeling change.
- **Multi-modal model:** incorporate processing conditions (anneal temperature/time,
  cooling method) alongside the image as additional model inputs, since these are
  physically linked to the resulting microstructure.

---

## Acknowledgments

Dataset: UHCS Ultra-High Carbon Steel micrograph dataset (Hecht, DeCost, Holm et al.,
Carnegie Mellon University / NIST).

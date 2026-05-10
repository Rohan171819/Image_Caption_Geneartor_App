<div align="center">

# 🖼️ Image Captioning with CNN + LSTM

### Encoder–Decoder Architecture for Automatic Image Description

> _Give a model an image. Get back a sentence. That's the magic._

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep_Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![NumPy](https://img.shields.io/badge/NumPy-Data-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-3B7A57?style=for-the-badge&logo=python&logoColor=white)](https://nltk.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Research](https://img.shields.io/badge/Research-Published_Paper-8B0000?style=for-the-badge&logo=academia&logoColor=white)](#-research)

</div>

---

## 📌 What It Does

This project implements an **encoder–decoder image captioning system** that automatically generates natural language descriptions of images — bridging Computer Vision and Natural Language Processing in a single end-to-end pipeline.

Given any image as input, the model produces a grammatically coherent, contextually accurate caption. It was trained on the **Flickr8k dataset** and evaluated using **BLEU scores** — the standard metric for caption quality.

### 🎯 Core Capabilities

| Capability | Detail |
|---|---|
| 🏗️ **CNN Encoder** | VGG16 (ImageNet pretrained) extracts a 4096-dim feature vector per image |
| 📝 **LSTM Decoder** | Recurrent decoder generates captions word-by-word from visual features |
| 🔤 **Word Embeddings** | Vocabulary-mapped token sequences with `<start>` / `<end>` delimiters |
| 📊 **BLEU Evaluation** | BLEU-1 and BLEU-2 scoring via NLTK for caption quality measurement |
| 🖼️ **Inference API** | Single image → caption in one function call |
| 📄 **Published Research** | Architecture and findings documented in a peer-reviewed paper |

---

## 🏛️ Architecture

### High-Level Pipeline

```
 ┌──────────────┐     ┌─────────────────────────────────────────────────┐
 │  INPUT IMAGE │────▶│                  CNN ENCODER                    │
 │  (224×224×3) │     │                                                 │
 └──────────────┘     │   Conv2D → Pool → Conv2D → Pool → ... (×5)     │
                       │                                                 │
                       │         VGG16 (ImageNet Pretrained)             │
                       │         FC Layer → 4096-dim Feature Vector      │
                       └───────────────────┬─────────────────────────────┘
                                           │
                                    image_features
                                    (4096,) vector
                                           │
                       ┌───────────────────▼─────────────────────────────┐
                       │               LSTM DECODER                       │
                       │                                                  │
                       │  <start>                                         │
                       │     │                                            │
                       │  [Embedding] ← word_index                       │
                       │     │                                            │
                       │  [Add] ← image_features (Dense → 256-dim)       │
                       │     │                                            │
                       │  [LSTM: 256 units]                               │
                       │     │                                            │
                       │  [Dense → vocab_size] → softmax                 │
                       │     │                                            │
                       │  next_word → feed back → repeat                 │
                       │     │                                            │
                       │  <end>  ← stop condition                        │
                       └───────────────────┬──────────────────────────────┘
                                           │
                                           ▼
                            "a dog running through a field"
```

### Detailed Model Graph

```
IMAGE INPUT          SEQUENCE INPUT
    │                     │
[VGG16 CNN]          [Embedding Layer]
    │                     │
[Dense 256]          [Dropout 0.5]
    │                     │
    └──────[Add]──────────┘
               │
          [LSTM 256]
               │
          [Dense 256]
               │
          [ReLU Activation]
               │
          [Dense vocab_size]
               │
          [Softmax]
               │
          PREDICTED WORD
```

### Training Data Flow

```
Flickr8k Dataset
      │
      ├── images/          (8,091 images)
      └── captions.txt     (5 captions × 8,091 = 40,455 total)
            │
            ▼
    ┌──────────────────┐
    │  Preprocessing   │
    │  • Lowercase     │
    │  • Remove punct  │
    │  • Add <start>   │
    │    <end> tokens  │
    └────────┬─────────┘
             │
    ┌────────▼──────────┐     ┌──────────────────────┐
    │  Tokenizer        │     │  VGG16 Feature        │
    │  (vocab ~8000)    │     │  Extraction           │
    │  word ↔ index     │     │  image → (4096,)      │
    └────────┬──────────┘     └──────────┬────────────┘
             │                            │
             └──────────[Merge]───────────┘
                              │
                       [Encoder-Decoder]
                              │
                      [BLEU Evaluation]
```

---

## 📊 Results

| Metric | Score |
|---|---|
| **BLEU-1** | 0.57 |
| **BLEU-2** | 0.33 |
| **Vocab Size** | ~8,000 tokens |
| **Training Images** | 6,000 |
| **Validation Images** | 1,000 |
| **Test Images** | 1,091 |
| **Epochs** | 20 |
| **Embedding Dim** | 256 |
| **LSTM Units** | 256 |

---

## 🖥️ Demo

> **Replace this section** with actual output screenshots once you run inference on test images.

```
📹  Suggested demo format:
    1. Pick 4–5 visually diverse test images (dog, crowd, food, sport, nature)
    2. Run: python predict.py --image test_images/dog.jpg
    3. Screenshot the image + generated caption side by side
    4. Compile into a grid (use Matplotlib's subplot grid)
    5. Save as demo.png and drop it here
```

**Example output format (recreate with real images):**

```
┌────────────────────────────────────────────────────────┐
│                                                        │
│  [Image: dog running on grass]                         │
│  ✅ Caption: "a brown dog is running through           │
│              a grassy field"                           │
│                                                        │
│  [Image: two people playing guitar]                    │
│  ✅ Caption: "two people are playing musical           │
│              instruments on a stage"                   │
│                                                        │
│  [Image: child on bicycle]                             │
│  ✅ Caption: "a young child is riding a bicycle        │
│              on a path"                                │
│                                                        │
│  BLEU-1: 0.57  │  BLEU-2: 0.33  │  Vocab: 8,091      │
└────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **CNN Encoder** | VGG16 (Keras Applications) | Image feature extraction (4096-dim) |
| **RNN Decoder** | LSTM (Keras) | Sequential caption generation |
| **DL Framework** | TensorFlow 2.x + Keras | Model definition, training, inference |
| **Embeddings** | Keras Embedding Layer | Word → dense vector mapping |
| **NLP / Eval** | NLTK | Tokenization + BLEU score computation |
| **Data** | NumPy + Pandas | Feature arrays & caption preprocessing |
| **Visualization** | Matplotlib | Loss curves + caption overlays |
| **Dataset** | Flickr8k | 8,091 images · 5 captions each |
| **Notebook** | Jupyter | Training + exploration environment |

---

## ⚡ How to Run

### Prerequisites

- Python 3.9+
- pip
- ~2GB disk space (Flickr8k + VGG16 weights)
- GPU recommended (CPU works, slower training)

### 1. Clone & Install

```bash
git clone https://github.com/your-username/image-captioning.git
cd image-captioning

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Download the Dataset

```bash
# Flickr8k — request access at:
# https://illinois.edu/fb/sec/1713398

# Place files as:
# data/
# ├── Images/          (8,091 .jpg files)
# └── captions.txt

mkdir -p data
# Move your downloaded files into data/
```

### 3. Extract CNN Features (one-time)

```bash
python src/extract_features.py
# Runs all 8,091 images through VGG16
# Saves: data/features.pkl  (~300MB)
# Time: ~10 min on CPU, ~2 min on GPU
```

### 4. Train the Model

```bash
python src/train.py \
  --epochs 20 \
  --batch_size 32 \
  --embedding_dim 256 \
  --lstm_units 256

# Checkpoints saved to: models/
# Training log saved to: logs/training.csv
```

### 5. Evaluate (BLEU Score)

```bash
python src/evaluate.py \
  --model models/model_epoch_20.h5 \
  --features data/features.pkl

# Output:
# BLEU-1: 0.57
# BLEU-2: 0.33
```

### 6. Run Inference on Your Image

```bash
python predict.py --image path/to/your/image.jpg

# Output:
# Caption: "a dog is running through a grassy field"
```

### Jupyter Notebook (Quick Start)

```bash
jupyter notebook notebooks/image_captioning.ipynb
```
Step through cells sequentially — feature extraction → training → evaluation → demo.

---

## 📁 Project Structure

```
image-captioning/
├── src/
│   ├── extract_features.py    # VGG16 feature extraction for all images
│   ├── preprocess.py          # Caption cleaning, tokenization, vocab build
│   ├── model.py               # Encoder-decoder model definition
│   ├── train.py               # Training loop with callbacks
│   ├── evaluate.py            # BLEU-1 and BLEU-2 scoring
│   └── utils.py               # Helper functions
├── notebooks/
│   └── image_captioning.ipynb # Full walkthrough notebook
├── models/                    # Saved .h5 model checkpoints
├── data/                      # Dataset (not committed — see setup)
├── test_images/               # Sample images for inference demo
├── predict.py                 # Single-image inference script
├── requirements.txt
└── README.md
```

---

## 📦 requirements.txt

```
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
nltk>=3.7
Pillow>=9.0.0
tqdm>=4.64.0
jupyter>=1.0.0
```

---

## 📄 Research

This project is backed by a **published research paper** documenting the architecture design, dataset preparation, training strategy, and BLEU evaluation methodology.

> 📎 **[Link to paper — add your DOI or journal link here]**

**Citation:**
```bibtex
@article{sharma2024imagecaptioning,
  title   = {Image Captioning using CNN and LSTM Encoder-Decoder Architecture},
  author  = {Sharma, Rohan},
  year    = {2024},
  journal = {[Journal/Conference Name]},
  url     = {[DOI link]}
}
```

---

## 🗺️ Roadmap

- [x] VGG16 encoder + LSTM decoder baseline
- [x] Flickr8k training + BLEU evaluation
- [x] Published research paper
- [ ] Attention mechanism (Bahdanau / Visual Attention)
- [ ] Beam search decoding (vs greedy)
- [ ] MSCOCO dataset fine-tuning
- [ ] Streamlit web app for live demo

---

## 👤 Author

**Rohan Sharma** — AI/ML Engineer · LangGraph Developer  
MCA @ GL Bajaj Institute of Technology & Management

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/rohan-sharma-048266246)
[![Email](https://img.shields.io/badge/Email-sharma1718rohan@gmail.com-D14836?style=flat-square&logo=gmail)](mailto:sharma1718rohan@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/your-username)

---

<div align="center">

**⭐ Star this repo if the captions made you smile**

</div>

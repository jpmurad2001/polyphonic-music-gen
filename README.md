# Polyphonic Music Generation Pipeline 🎹

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

> **Project Goal:** A comparative study between **LSTM (Long Short-Term Memory)** and **Transformer** architectures for symbolic music generation, addressing the *Vanishing Gradient* problem in long musical sequences.

*[Developed as a Capstone Engineering Project at UFF - 2025]*

---

## 🚀 Key Features & Results

This project benchmarks two Deep Learning approaches to generate piano music based on the **MAESTRO** and **Chopin** datasets.

| Metric | LSTM Model (Baseline) | Transformer Model (Proposed) |
| :--- | :--- | :--- |
| **Architecture** | Recurrent Neural Network (RNN) | Attention-Based Mechanism |
| **Long-Term Memory** | Fails after ~20s (loses rhythm) | **High coherence** (retains motif) |
| **Training Efficiency** | Fast convergence | Computationally intensive (High VRAM) |
| **Musicality** | Good for short phrases | **Superior structural complexity** |

---

## 🛠️ Tech Stack & Engineering Challenges

| Category | Tools / Techniques |
| :--- | :--- |
| **Deep Learning** | `PyTorch`, `TensorFlow/Keras`, `Attention Mechanisms` |
| **Audio Processing** | `pretty_midi`, `music21`, `librosa`, `FluidSynth` |
| **Infrastructure** | Google Colab Pro (T4/V100 GPUs) |
| **Optimization** | Solved **OOM (Out of Memory)** errors by implementing custom batching strategies and optimizing sequence lengths for VRAM constraints. |

---

## 📂 Project Structure

The core logic is implemented in Jupyter Notebooks using **PyTorch**.

```text
├── notebooks/
│   ├── transformer_maestro_v2.ipynb  <-- ⭐ MAIN MODEL (Best Results)
│   ├── lstm_maestro_v2.ipynb         <-- Baseline Model
│   ├── midi_preprocessing.ipynb      <-- Data Engineering Pipeline
│   └── (legacy_experiments)/         <-- Older V1 iterations
│
├── docs/
│   ├── Project_Report_PT.pdf         <-- Original Thesis (Portuguese)
│   └── Presentation_PT.pdf           <-- Defense Slides
│
├── generated_samples/                <-- Listen to the AI output here (.wav/.mid)
└── requirements.txt
---
```
## 🎹 How to Run (Reproduction)
Due to the size of the MAESTRO dataset (~GBs), data is not included in this repo.

1. Clone the repo

```

git clone https://github.com/SEU_USUARIO/polyphonic-music-gen.git

```

2. Download the Dataset

• Get the MAESTRO (MIDI only) dataset from Magenta TensorFlow(https://magenta.tensorflow.org/datasets/maestro).

• Unzip it into a folder named `data/` in the root directory.

3. Install Dependencies

```

pip install -r requirements.txt

```

4. Train

Open `notebooks/transformer_maestro_v2.ipynb` via Jupyter Lab or Google Colab and run the cells.

---

📜 Documentation
• Read the Full Thesis (PT-BR)(docs/Project_Report_PT.pdf)

• (English translation coming soon)

---

**Author:** João Pedro Murad
*Production Engineer | Data Scientist | AI Enthusiast*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](www.linkedin.com/joãopedrosmurad)

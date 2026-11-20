# EEG motor imagery: from raw signals to a hyperparameter-tuned network 🧠

Practical session of **Artificial Intelligence in Neurotechnology** (Master in Neurotechnology, Universidad Politécnica de Madrid).

Can a computer tell whether you are imagining moving your **left** or your **right** hand, only from your EEG, if it has **never seen you before**? In this session you build the whole pipeline step by step: look at the data, clean it, find the brain signature, catch a model cheating, build a classic BCI, train neural networks and tune them with **KerasTuner**.

You will not write code: in every step you **predict**, **decide** with the controls of each cell, **run**, and **interpret** the figures.

## Before the class ✅
1. You need a Google account.
2. Open **00 · Setup check** below, then **File → Save a copy in Drive**.
3. **Runtime → Change runtime type → T4 GPU → Save**, then **Runtime → Run all**.
4. You should see **🎉 All good** at the bottom.

## Notebooks

| | Notebook | What you do |
|---|---|---|
| 00 | [Setup check](https://colab.research.google.com/github/ptorija/eeg-motor-imagery-practical/blob/main/notebooks/00_setup_check.ipynb) | Check that everything works (at home) |
| 01 | [Meet the data](https://colab.research.google.com/github/ptorija/eeg-motor-imagery-practical/blob/main/notebooks/01_meet_the_data.ipynb) | The experiment, the electrodes, a first look at the signals |
| 02 | [Preprocessing](https://colab.research.google.com/github/ptorija/eeg-motor-imagery-practical/blob/main/notebooks/02_preprocessing.ipynb) | Filters, artifacts, standardisation |
| 03 | [Brain… or eyes?](https://colab.research.google.com/github/ptorija/eeg-motor-imagery-practical/blob/main/notebooks/03_brain_signature.ipynb) | The ERD signature, and a model that cheats with eye movements |
| 04 | [Split & baseline](https://colab.research.google.com/github/ptorija/eeg-motor-imagery-practical/blob/main/notebooks/04_split_and_baseline.ipynb) | Testing on new people, CSP + LDA, permutation check |
| 05 | [Neural networks](https://colab.research.google.com/github/ptorija/eeg-motor-imagery-practical/blob/main/notebooks/05_neural_networks.ipynb) | MLP, CNN, RNN/LSTM/GRU, BiGRU, attention, EEGNet |
| 06 | [Hyperparameter search](https://colab.research.google.com/github/ptorija/eeg-motor-imagery-practical/blob/main/notebooks/06_hyperparameter_search.ipynb) | KerasTuner on a recurrent network and on EEGNet |
| 07 | [Final test](https://colab.research.google.com/github/ptorija/eeg-motor-imagery-practical/blob/main/notebooks/07_final_test.ipynb) | Open the test set once, results per person |

Working in a group or alone is up to you.

## Running locally instead of Colab
Python 3.10–3.12:
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```
Then open the notebooks from the `notebooks/` folder. The dataset (~180 MB) is downloaded automatically the first time. If something fails locally, use Colab.

## Data and credits
- **Dataset:** PhysioNet *EEG Motor Movement/Imagery Dataset* (v1.0.0), Open Data Commons Attribution License v1.0. We use the imagined left/right fist runs (4, 8, 12) of 106 subjects (S088, S092 and S100 excluded: different sampling rate), 64 channels, resampled to 80 Hz, 4-s trials from cue onset. The preprocessed file is in the [releases](https://github.com/ptorija/eeg-motor-imagery-practical/releases).
  - Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004). BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. *IEEE Transactions on Biomedical Engineering* 51(6):1034–1043.
  - Goldberger, A. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220.
- **EEGNet:** Lawhern, V.J. et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering* 15:056013.
- **Tools:** MNE-Python, Keras, KerasTuner, scikit-learn.

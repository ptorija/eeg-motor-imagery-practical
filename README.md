# EEG motor imagery: from raw signals to a hyperparameter-tuned network 🧠

Practical session of **Artificial Intelligence in Neurotechnology** (Master in Neurotechnology, Universidad Politécnica de Madrid).

Can a computer tell whether you are imagining moving your **left** or your **right** hand, only from your EEG, if it has **never seen you before**? In this session you build the whole pipeline step by step: look at the data, clean it, find the brain signature, catch a model cheating, build a classic BCI, train neural networks and tune them with **KerasTuner**.

You will not write code: in every step you **predict**, **decide** with the controls of each cell, **run**, and **interpret** the figures.

## 👉 [Open the notebook in Colab](https://colab.research.google.com/github/ptorija/eeg-motor-imagery-practical/blob/main/notebooks/EEG_motor_imagery_practical.ipynb)

Everything is in **one notebook**, split in 8 folded parts. Open each part (▸ next to its title) only when we get there.

## Before the class ✅
1. You need a Google account.
2. Open the notebook above, then **File → Save a copy in Drive**.
3. Check **Runtime → Change runtime type → T4 GPU** (the notebook asks for it by default).
4. Run the cell of **part 00 · Setup**. You should see **✅ Ready**.

## The parts

| | Part | What you do |
|---|---|---|
| 00 | Setup | Install, download the data, check the GPU |
| 01 | Meet the data | The experiment, the electrodes, a first look at the signals |
| 02 | Preprocessing | Filters (up to 75 Hz: spot the 60 Hz power line), artifacts, standardisation |
| 03 | Brain… or eyes? | The ERD signature, and a model that cheats with eye movements |
| 04 | Split & baseline | Testing on new people, CSP + LDA, permutation check |
| 05 | Neural networks | MLP, CNN, RNN/LSTM/GRU, BiGRU, attention, EEGNet |
| 06 | Hyperparameter search | KerasTuner on a recurrent network and on EEGNet |
| 07 | Final test | Open the test set once, results per person |

Working in a group or alone is up to you. If you arrive late or Colab disconnects, click the first cell of the current part and press **Ctrl + F8** (*Run before*).

📊 **Live class poll.** Your predictions, choices and 🏆 results are sent anonymously (random nickname, no name or email) to a Google Form that the teacher projects. You can switch it off in the setup cell (`share_answers`).

## Running locally instead of Colab
Python 3.10–3.12:
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```
Then open `notebooks/EEG_motor_imagery_practical.ipynb`. The dataset (~215 MB) is downloaded automatically the first time. If something fails locally, use Colab.

## Data and credits
- **Dataset:** PhysioNet *EEG Motor Movement/Imagery Dataset* (v1.0.0), Open Data Commons Attribution License v1.0. We use the imagined left/right fist runs (4, 8, 12) of 106 subjects (S088, S092 and S100 excluded: different sampling rate), 64 channels, 160 Hz as recorded, 4-s trials from cue onset (built with `scripts/make_dataset_160Hz.py`). The preprocessed file is in the [releases](https://github.com/ptorija/eeg-motor-imagery-practical/releases).
  - Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004). BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. *IEEE Transactions on Biomedical Engineering* 51(6):1034–1043.
  - Goldberger, A. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220.
- **EEGNet:** Lawhern, V.J. et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering* 15:056013.
- **Tools:** MNE-Python, Keras, KerasTuner, scikit-learn.

"""
bci_utils.py — helper functions for the EEG motor-imagery practical.

Everything that is *not* the point of the exercise (downloading, plotting
style, boilerplate model code) lives here, so the notebooks can focus on the
decisions. You are welcome to read it: nothing here is hidden magic.
"""
import os
import json
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch

REPO = "ptorija/eeg-motor-imagery-practical"
DATA_URL = f"https://github.com/{REPO}/releases/download/v1.0/eegmmidb_MI_LR.npz"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/main"
DATA_FILE = "eegmmidb_MI_LR.npz"

# One colour per class, used in every figure of the course
COLORS = {"left": "#2E6FD8", "right": "#F28E2B"}
CLASS_NAMES = ["left", "right"]          # label 0 = left hand, 1 = right hand
GREY = "#8A8A8A"


# ----------------------------------------------------------------------------
# Style & data
# ----------------------------------------------------------------------------
def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110, "figure.figsize": (9, 4),
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "font.size": 11,
        "axes.titleweight": "bold", "legend.frameon": False,
    })


def load_data(path=DATA_FILE, verbose=True):
    """Download (only the first time) and load the dataset.

    Returns a dict with:
      X        (n_trials, 64 channels, 320 samples)  EEG in microvolts
      y        (n_trials,)  0 = imagined LEFT hand, 1 = imagined RIGHT hand
      subject  (n_trials,)  subject id (1..109)
      ch_names, ch_pos, fs (=80 Hz)
    """
    if not os.path.exists(path):
        if verbose:
            print("Downloading the dataset (~180 MB, once)...")
        urllib.request.urlretrieve(DATA_URL, path)
    d = np.load(path)
    data = {
        "X": d["X"].astype(np.float32),
        "y": (d["y"] == "right").astype("int32"),
        "subject": d["subject"].astype(int),
        "run": d["run"].astype(int),
        "ch_names": [str(c) for c in d["ch_names"]],
        "ch_pos": d["ch_pos"],
        "fs": int(d["fs"]),
    }
    if verbose:
        n, c, t = data["X"].shape
        print(f"Loaded {n} trials | {c} channels | {t} samples "
              f"({t / data['fs']:.0f} s at {data['fs']} Hz) | "
              f"{len(np.unique(data['subject']))} subjects")
    return data


def mne_info(data):
    """An mne.Info object with the electrode positions (for MNE plots)."""
    import mne
    info = mne.create_info(data["ch_names"], data["fs"], ch_types="eeg")
    montage = mne.channels.make_dig_montage(
        ch_pos=dict(zip(data["ch_names"], data["ch_pos"])), coord_frame="head")
    info.set_montage(montage)
    return info


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def bandpass(X, low, high, fs=80, order=4):
    """Zero-phase Butterworth band-pass filter along the time axis."""
    high = min(high, fs / 2 - 1)
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def peak_to_peak(X):
    """Largest peak-to-peak amplitude (µV) of each trial, over all channels."""
    return (X.max(-1) - X.min(-1)).max(-1)


def standardize_trials(X):
    """Each channel of each trial -> mean 0, std 1.

    Uses only the trial itself, so it cannot leak information between
    training and test sets."""
    X = (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-6)
    return np.clip(X, -10, 10).astype(np.float32)


def psd(X, fs=80):
    """Power spectral density (Welch, 1-s windows). Returns freqs, power."""
    f, P = welch(X, fs=fs, nperseg=fs, axis=-1)
    return f, P


# ----------------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------------
def subject_split(subject, n_test=21, n_val=15, seed=0):
    """Split by SUBJECT: every person is in exactly one of train / val / test."""
    rng = np.random.RandomState(seed)
    subs = rng.permutation(np.unique(subject))
    test_s, val_s, train_s = subs[:n_test], subs[n_test:n_test + n_val], subs[n_test + n_val:]
    tr, va, te = (np.isin(subject, s) for s in (train_s, val_s, test_s))
    # Safety checks: no person appears in two sets
    assert not set(train_s) & set(val_s) and not set(train_s) & set(test_s) and not set(val_s) & set(test_s)
    assert (tr.astype(int) + va + te == 1).all()
    return tr, va, te


def trial_split(n, test_fraction, seed=0):
    """Split by TRIAL at random (ignores who the trial belongs to)."""
    rng = np.random.RandomState(seed)
    te = np.zeros(n, bool)
    te[rng.permutation(n)[:int(round(n * test_fraction))]] = True
    return ~te, te


def chance_level(y):
    """Accuracy of always predicting the most frequent class."""
    return max(np.mean(y), 1 - np.mean(y))


def spectral_features(Xf, fs=80, fmin=6, fmax=32):
    """Log power spectrum (1-Hz bins, fmin..fmax) of every electrode, flattened."""
    f, P = psd(Xf, fs)
    sel = (f >= fmin) & (f <= fmax)
    return np.log(P[..., sel] + 1e-12).reshape(len(Xf), -1).astype(np.float32)


def band_power_features(Xf, fs=80, bands=((8, 13), (13, 30))):
    """Log power of each electrode in the mu (8-13 Hz) and beta (13-30 Hz) bands
    -> 64 electrodes x 2 bands = 128 numbers per trial (the ERD of notebook 03)."""
    f, P = psd(Xf, fs)
    B = np.stack([P[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in bands], -1)
    return np.log(B + 1e-12).reshape(len(Xf), -1).astype(np.float32)


def prepare_inputs(data, low=8, high=30, tmin=1.0):
    """The session's preprocessing (notebooks 02-03) + split (notebook 04):
    band-pass low-high Hz, drop the first `tmin` seconds (eye movements!),
    standardise each trial, split by person.

    Returns a dict with the masks train/val/test, the labels y and the inputs
    in the shape each model family expects:
      'features'   (n, 128)          mu & beta power of each electrode, for the MLP
      'sequence'   (n, time, 64)     time x channels, for CNN1D / RNNs / attention
      'image'      (n, 64, time, 1)  channels x time, for EEGNet
    """
    Xf = bandpass(data["X"], low, high, data["fs"])[:, :, int(round(tmin * data["fs"])):]
    train, val, test = subject_split(data["subject"])
    F = band_power_features(Xf, data["fs"])
    mu, sd = F[train].mean(0), F[train].std(0) + 1e-6     # scaling learnt on TRAIN people only
    Xs = standardize_trials(Xf)
    return {"train": train, "val": val, "test": test, "y": data["y"],
            "features": (F - mu) / sd,
            "sequence": np.ascontiguousarray(Xs.transpose(0, 2, 1)),
            "image": Xs[..., None]}


MODEL_INPUT = {"MLP (band power)": "features", "CNN1D": "sequence", "SimpleRNN": "sequence",
               "LSTM": "sequence", "GRU": "sequence", "BiGRU": "sequence",
               "Attention": "sequence", "EEGNet": "image"}


def build_model(name, inp, rnn_summary="average"):
    """Build one of the models of notebook 05 by name, sized for the inputs `inp`."""
    n_t, n_c = inp["sequence"].shape[1:]
    if name.startswith("MLP"):
        return build_mlp(inp["features"].shape[1])
    if name == "CNN1D":
        return build_cnn1d(n_t, n_c)
    if name in ("SimpleRNN", "LSTM", "GRU"):
        return build_rnn(name, summary=rnn_summary, n_times=n_t, n_chans=n_c)
    if name == "BiGRU":
        return build_rnn("GRU", bidirectional=True, summary=rnn_summary, n_times=n_t, n_chans=n_c)
    if name == "Attention":
        return build_attention(n_t, n_c)
    if name == "EEGNet":
        return build_eegnet(n_c, n_t)
    raise ValueError(name)


# ----------------------------------------------------------------------------
# Models (Keras)
# ----------------------------------------------------------------------------
def _compile(model, lr=1e-3):
    import keras
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _front_end(x, filters=32):
    """Shared first block for the sequence models: a temporal convolution
    that also shrinks 320 time steps to 80 (keeps training fast)."""
    from keras import layers
    x = layers.Conv1D(filters, 9, padding="same", activation="elu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling1D(4)(x)
    return layers.Dropout(0.3)(x)


def build_mlp(n_features, units=64, dropout=0.5, lr=1e-3):
    """One hidden layer (more layers overfit with so little data — we tried)."""
    import keras
    from keras import layers
    i = keras.Input((n_features,))
    x = layers.Dense(units, activation="relu")(i)
    x = layers.Dropout(dropout)(x)
    return _compile(keras.Model(i, layers.Dense(2, activation="softmax")(x), name="MLP"), lr)


def build_cnn1d(n_times=320, n_chans=64, filters=32, lr=1e-3):
    import keras
    from keras import layers
    i = keras.Input((n_times, n_chans))
    x = _front_end(i, filters)
    x = layers.Conv1D(filters, 9, padding="same", activation="elu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    return _compile(keras.Model(i, layers.Dense(2, activation="softmax")(x), name="CNN1D"), lr)


def build_rnn(kind="GRU", bidirectional=False, units=32, summary="average",
              n_times=320, n_chans=64, lr=1e-3):
    """kind = 'SimpleRNN', 'LSTM' or 'GRU'.
    summary = 'last'    -> the classifier only sees the final hidden state
              'average' -> the classifier sees the average of the hidden states
                           of all time steps"""
    import keras
    from keras import layers
    i = keras.Input((n_times, n_chans))
    x = _front_end(i)
    rnn = getattr(layers, kind)(units, return_sequences=(summary == "average"))
    x = layers.Bidirectional(rnn)(x) if bidirectional else rnn(x)
    if summary == "average":
        x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    name = ("Bi" if bidirectional else "") + kind
    return _compile(keras.Model(i, layers.Dense(2, activation="softmax")(x), name=name), lr)


def build_attention(n_times=320, n_chans=64, heads=4, lr=1e-3):
    import keras
    from keras import layers
    i = keras.Input((n_times, n_chans))
    x = _front_end(i)
    x = layers.AveragePooling1D(2)(x)
    a = layers.MultiHeadAttention(num_heads=heads, key_dim=8, dropout=0.3)(x, x)
    x = layers.LayerNormalization()(x + a)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    return _compile(keras.Model(i, layers.Dense(2, activation="softmax")(x), name="Attention"), lr)


def build_eegnet(n_chans=64, n_times=320, F1=8, D=2, kernel=40, dropout=0.5, lr=1e-3):
    """EEGNet (Lawhern et al., 2018), input shape (channels, time, 1)."""
    import keras
    from keras import layers
    i = keras.Input((n_chans, n_times, 1))
    x = layers.Conv2D(F1, (1, kernel), padding="same", use_bias=False)(i)        # temporal filters
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D((n_chans, 1), depth_multiplier=D, use_bias=False)(x)  # spatial filters
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.SeparableConv2D(F1 * D, (1, 16), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Flatten()(x)
    return _compile(keras.Model(i, layers.Dense(2, activation="softmax")(x), name="EEGNet"), lr)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_history(history, title="", chance=0.5, ax=None):
    """Training vs validation accuracy per epoch."""
    h = history.history if hasattr(history, "history") else history
    ax = ax or plt.subplots(figsize=(7, 3.5))[1]
    ep = np.arange(1, len(h["accuracy"]) + 1)
    ax.plot(ep, h["accuracy"], "o-", ms=3, label="training")
    ax.plot(ep, h["val_accuracy"], "o-", ms=3, label="validation (new people)")
    ax.axhline(chance, color=GREY, ls="--", lw=1, label="chance")
    top = max(0.85, max(max(h["accuracy"]), max(h["val_accuracy"])) + 0.05)
    ax.set(xlabel="epoch", ylabel="accuracy", ylim=(0.4, min(1.0, top)), title=title)
    ax.legend(loc="upper left")
    return ax


def plot_confusion(y_true, y_pred, title="", ax=None):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pct = cm / cm.sum(1, keepdims=True) * 100
    ax = ax or plt.subplots(figsize=(3.6, 3.2))[1]
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{pct[i, j]:.0f}%\n({cm[i, j]})", ha="center", va="center",
                    color="white" if pct[i, j] > 60 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["left", "right"],
           yticklabels=["left", "right"], xlabel="predicted", ylabel="true", title=title)
    ax.grid(False)
    return ax


def plot_scores(names, scores, chance=0.5, title="", errors=None, ax=None, highlight=None):
    """Horizontal bar chart of accuracies with the chance line."""
    ax = ax or plt.subplots(figsize=(7, 0.45 * len(names) + 1))[1]
    colors = ["#4C72B0" if (highlight is None or n == highlight) else "#B8C6DB" for n in names]
    ax.barh(names, scores, xerr=errors, color=colors, capsize=3)
    for k, s in enumerate(scores):
        ax.text(s + 0.01, k, f"{s:.2f}", va="center")
    ax.axvline(chance, color=GREY, ls="--", lw=1)
    ax.set(xlim=(0.4, 1.0), xlabel="accuracy")
    ax.set_title(title, pad=16)
    ax.invert_yaxis()
    ax.text(chance, 1.01, "chance", color=GREY, ha="center", va="bottom", transform=ax.get_xaxis_transform())
    return ax


def load_precomputed(name):
    """Results computed in advance (plan B if training is too slow)."""
    path = f"precomputed/{name}.json"
    if not os.path.exists(path):
        os.makedirs("precomputed", exist_ok=True)
        urllib.request.urlretrieve(f"{RAW_URL}/precomputed/{name}.json", path)
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))

"""Build eegmmidb_MI_LR_160Hz.npz from the PhysioNet zip (instructor only).

Usage: python make_dataset_160Hz.py eegmmidb-1.0.0.zip eegmmidb_MI_LR.npz out.npz
  - eegmmidb-1.0.0.zip : PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0
  - eegmmidb_MI_LR.npz : the previous 80 Hz release (v1.0), used only for the
                         electrode positions and to check that trials match
Imagined left/right fist runs (4, 8, 12), 106 subjects (S088, S092, S100 have a
different sampling rate), 4-s trials from cue onset, 160 Hz, microvolts, float16.
"""
import os, sys, tempfile, zipfile
import numpy as np
import mne

mne.set_log_level("ERROR")
src, old_path, out = sys.argv[1:4]
old = np.load(old_path)
tmp = tempfile.mkdtemp()
z = zipfile.ZipFile(src)
Xs, ys, ss, rs = [], [], [], []
for s in range(1, 110):
    if s in (88, 92, 100):
        continue
    for r in (4, 8, 12):
        p = z.extract(f"files/S{s:03d}/S{s:03d}R{r:02d}.edf", tmp)
        raw = mne.io.read_raw_edf(p, preload=True)
        assert raw.info["sfreq"] == 160, (s, r, raw.info["sfreq"])
        raw.rename_channels(lambda c: c.strip(".").upper().replace("Z", "z").replace("FP", "Fp"))
        ev, ed = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2})
        ep = mne.Epochs(raw, ev, ed, tmin=0, tmax=4 - 1 / 160, baseline=None, preload=True)
        Xs.append(ep.get_data() * 1e6)
        ys += ["left" if e == 1 else "right" for e in ep.events[:, 2]]
        ss += [s] * len(ep)
        rs += [r] * len(ep)
        os.remove(p)
X = np.concatenate(Xs).astype(np.float16)
names = list(ep.ch_names)
assert names == list(old["ch_names"]) and (np.array(ys) == old["y"]).all() and (np.array(ss) == old["subject"]).all()
np.savez_compressed(out, X=X, y=np.array(ys), subject=np.array(ss, np.int16), run=np.array(rs, np.int8),
                    ch_names=np.array(names), ch_pos=old["ch_pos"], fs=np.int16(160))
print(X.shape, "->", out)

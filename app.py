###############################################################################
# Streamlit GUI for Dyslexia-EEG CNN (EDF or FIF upload) – FULL FILE
###############################################################################
import sys, pathlib, tempfile, re, warnings

import mne
import numpy as np
import streamlit as st
import torch

# ------------------------------------------------------------------ repo path
ROOT = pathlib.Path(__file__).parent
sys.path.append(str(ROOT / "src"))  # enable  from src.*  imports

from src.train import Lit
from src.models.cnn import SimpleCNN  # only for channel inspection

# ------------------------------------------------------------------ settings
CKPT_DIR = ROOT / "checkpoints"
DEVICE   = "cpu"                    # set "cuda" if you have a GPU wheel

# ------------------------------------------------------------------ find ckpt
def val_score(p: pathlib.Path) -> float:
    m = re.search(r"val[_=]?([0-9.]+)", p.stem)
    return float(m.group(1)) if m else -1.0

ckpts = list(CKPT_DIR.glob("*.ckpt"))
if not ckpts:
    st.error("No checkpoints in ./checkpoints – train the model first!")
    st.stop()

BEST_CKPT = max(ckpts, key=val_score)
st.sidebar.info(f"Loaded checkpoint: {BEST_CKPT.name}")

# ------------------------------------------------------------------ load model
@st.cache_resource(hash_funcs={"torch.nn.modules.module.Module": id})
def load_model(path: pathlib.Path):
    lit = Lit.load_from_checkpoint(path, map_location=DEVICE)
    model = lit.model.to(DEVICE).eval()
    n_ch  = model.net[0].in_channels
    return model, n_ch

MODEL, N_CHANNELS = load_model(BEST_CKPT)

# ------------------------------------------------------------------ helpers
def preprocess_to_windows(raw: mne.io.BaseRaw, win_sec=3, step_sec=3):
    """band-pass 1-40 Hz, z-score per channel, return (n_win, C, T)"""
    raw.filter(1., 40., verbose="ERROR")
    data = raw.get_data()
    data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
    sf   = raw.info["sfreq"]
    win, step = int(win_sec*sf), int(step_sec*sf)
    xs = [data[:, i:i+win] for i in range(0, data.shape[1]-win+1, step)]
    return np.stack(xs)

# ------------------------------------------------------------------ UI
st.title("EEG Dyslexia Detector")
st.subheader("Upload a single-subject **.edf** or **.fif** file")

up = st.file_uploader("File", type=["edf","fif"])

if up:
    with st.spinner("Reading & preprocessing…"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix="."+up.name.split(".")[-1])
        tmp.write(up.getbuffer()); tmp.close()
        if up.name.lower().endswith(".edf"):
            raw = mne.io.read_raw_edf(tmp.name, preload=True, verbose="ERROR")
        else:
            raw = mne.io.read_raw_fif(tmp.name, preload=True, verbose="ERROR")
        X = preprocess_to_windows(raw, win_sec=3, step_sec=3)
        pathlib.Path(tmp.name).unlink()

    st.success(f"Loaded • windows: {len(X)} • channels: {X.shape[1]}")
    if X.shape[1] != N_CHANNELS:
        st.error(f"Channel mismatch – model expects {N_CHANNELS}, file has {X.shape[1]}")
        st.stop()

    if st.button("Predict"):
        with st.spinner("Running inference…"):
            with torch.no_grad():
                logits = MODEL(torch.tensor(X, dtype=torch.float32).to(DEVICE))
                probs  = torch.softmax(logits,1)[:,1].cpu().numpy()
                preds  = (probs>=0.5).astype(int)

        st.subheader("Window-level P(Dyslexic)")
        st.line_chart(probs)

        vote = int(round(preds.mean()))
        label = "Dyslexic" if vote else "Normal"
        conf  = probs.mean()
        color = "red" if vote else "green"
        st.markdown(f"<h2 style='color:{color};'>Overall: {label}</h2>",
                    unsafe_allow_html=True)
        st.caption(f"Majority vote over {len(X)} windows • mean P = {conf:.1%}")

st.sidebar.markdown("---")
st.sidebar.write("Model : 1-D CNN, 4 conv layers")
st.sidebar.write("Preproc : 1-40 Hz BP, z-score, 3 s windows")
st.sidebar.write("Author : Raghav Sharma | 2025")

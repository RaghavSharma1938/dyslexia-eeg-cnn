###############################################################################
# Streamlit GUI – Dyslexia-EEG CNN  +  PDF Report
###############################################################################
import io, json, pathlib, re, sys, tempfile, time, datetime

import mne
import numpy as np
import streamlit as st
import torch
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ── repo path ───────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent
sys.path.append(str(ROOT / "src"))

from src.train import Lit                 # loads checkpoint

# ── settings ────────────────────────────────────────────────────────────────
CKPT_DIR = ROOT / "checkpoints"
DEVICE   = "cpu"

# ── find best checkpoint (largest val score in filename) ───────────────────
def _val_score(p): 
    m = re.search(r"val[_=]?([0-9.]+)", p.stem)
    return float(m.group(1)) if m else -1
ckpts = sorted(CKPT_DIR.glob("*.ckpt"), key=_val_score, reverse=True)
if not ckpts:
    st.error("No checkpoints found in ./checkpoints"); st.stop()
BEST_CKPT = ckpts[0]

# ── model loader (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    lit  = Lit.load_from_checkpoint(path, map_location=DEVICE)
    net  = lit.model.to(DEVICE).eval()
    n_ch = net.net[0].in_channels
    return net, int(n_ch)
MODEL, N_CH = load_model(BEST_CKPT)

# ── preprocessing helper ───────────────────────────────────────────────────
def preprocess(raw, win=3):
    raw.filter(1., 40., verbose="ERROR")
    d = raw.get_data()
    d = (d - d.mean(axis=1, keepdims=True)) / d.std(axis=1, keepdims=True)
    sf, step = raw.info["sfreq"], int(win * raw.info["sfreq"])
    return np.stack([d[:, i:i+step] for i in range(0, d.shape[1]-step+1, step)])

# ── PDF generator ──────────────────────────────────────────────────────────
def create_pdf(report_dict: dict) -> bytes:
    buf = io.BytesIO()
    c   = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, h-50, "Dyslexia EEG Report")

    c.setFont("Helvetica", 12)
    y = h-100
    for k, v in report_dict.items():
        c.drawString(40, y, f"{k}: {v}")
        y -= 22

    c.line(40, y-10, w-40, y-10)
    c.drawString(40, y-30, "Generated by Dyslexia-EEG CNN GUI")
    c.save()
    buf.seek(0)
    return buf.read()

# ── sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📦 Checkpoint")
    st.success(BEST_CKPT.name, icon="✅")
    st.markdown("---")
    st.markdown("**Model**: 1-D CNN (4 conv)  \n"
                "**Pre-proc**: 1-40 Hz BP, z-score, 3 s windows")
    st.caption("© Raghav Sharma · 2025")

# ── main UI ────────────────────────────────────────────────────────────────
st.title("🧠 Dyslexia Detector")
upl = st.file_uploader("Upload **.edf** or **.fif** file", type=["edf","fif"])

if upl:
    tmp = tempfile.NamedTemporaryFile(delete=False,
                                      suffix="."+upl.name.split(".")[-1])
    tmp.write(upl.getbuffer()); tmp.close()

    raw = (mne.io.read_raw_edf if upl.name.lower().endswith(".edf")
           else mne.io.read_raw_fif)(tmp.name, preload=True, verbose="ERROR")
    pathlib.Path(tmp.name).unlink()

    X = preprocess(raw)
    st.success(f"Loaded ✔  {len(X)} windows · {X.shape[1]} channels")

    if X.shape[1] != N_CH:
        st.error(f"Model expects {N_CH} channels, file has {X.shape[1]}"); st.stop()

    if st.button("Predict"):
        with st.spinner("Inferencing…"):
            logits = MODEL(torch.tensor(X, dtype=torch.float32).to(DEVICE))
            probs  = torch.softmax(logits, 1)[:, 1].detach().cpu().numpy()  # ← add .detach()
            preds  = (probs >= 0.5).astype(int)

        st.line_chart(probs)

        vote  = int(round(preds.mean()))
        label = "Dyslexic" if vote else "Normal"
        conf  = probs.mean()
        st.markdown(f"## Result: **{label}**  (confidence {conf:.2%})")

        # ─ create PDF & JSON
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        rep = dict(Prediction=label, Confidence=f"{conf:.2%}",
                   Windows=len(X), Channels=X.shape[1],
                   Checkpoint=BEST_CKPT.name, Time=timestamp)

        pdf_bytes = create_pdf(rep)
        st.download_button("Download PDF report", pdf_bytes,
                           file_name="dyslexia_report.pdf",
                           mime="application/pdf")

        st.download_button("Download JSON report",
                           json.dumps(rep, indent=2),
                           file_name="dyslexia_report.json",
                           mime="application/json")

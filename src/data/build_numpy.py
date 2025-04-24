import pathlib, argparse, numpy as np, mne

LABEL = {"h": 0, "s": 1}

def make_windows(fif_file: pathlib.Path, win_sec=2.0, step_sec=2.0):
    raw = mne.io.read_raw_fif(fif_file, preload=True, verbose="ERROR")
    data = raw.get_data()
    data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)

    sf = raw.info["sfreq"]
    win, step = int(win_sec*sf), int(step_sec*sf)
    xs = [data[:, i:i+win] for i in range(0, data.shape[1]-win+1, step)]
    return np.stack(xs)

def run(interim_dir="data/interim", out_dir="data/processed",
        win=2.0, step=2.0):
    interim_dir, out_dir = map(pathlib.Path, (interim_dir, out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    for fif in interim_dir.glob("*.fif"):
        X = make_windows(fif, win, step)
        y = np.full(len(X), LABEL[fif.stem[0]], np.int64)
        np.save(out_dir / f"{fif.stem}_X.npy", X)
        np.save(out_dir / f"{fif.stem}_y.npy", y)
        print(f"{fif.name:<8} → {len(X):4d} windows")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interim_dir", default="data/interim")
    p.add_argument("--out_dir",     default="data/processed")
    p.add_argument("--win",  type=float, default=2.0)
    p.add_argument("--step", type=float, default=2.0)
    run(**vars(p.parse_args()))

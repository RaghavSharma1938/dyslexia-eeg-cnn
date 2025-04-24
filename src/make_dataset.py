# minimal EDF → sliding-window tensors
import mne, numpy as np, argparse, pathlib

def windows_from_raw(raw, win_sec=2.0, step_sec=2.0):
    sf = raw.info["sfreq"]
    win, step = int(win_sec*sf), int(step_sec*sf)
    data = raw.get_data()
    xs = []
    for start in range(0, data.shape[1]-win+1, step):
        xs.append(data[:, start:start+win])
    return np.stack(xs)             # (n_win, n_chan, n_time)

def main(in_dir, out_dir):
    out_dir.mkdir(exist_ok=True, parents=True)
    for f in in_dir.glob("*.edf"):
        raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
        raw.filter(l_freq=1., h_freq=40., verbose="ERROR")
        X = windows_from_raw(raw)                  # tensor
        y = 0 if f.stem.startswith("h") else 1     # label
        np.save(out_dir/f"{f.stem}_X.npy", X)
        np.save(out_dir/f"{f.stem}_y.npy", np.full(len(X), y))
        print(f"✔ {f.name}: {X.shape[0]} windows")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir",  default="data/raw")
    p.add_argument("--out_dir", default="data/processed")
    args = p.parse_args()
    main(pathlib.Path(args.in_dir), pathlib.Path(args.out_dir))

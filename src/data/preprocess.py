import pathlib, argparse, mne

def edf_to_fif(edf_path: pathlib.Path, out_dir: pathlib.Path,
               l_freq=1.0, h_freq=40.0):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")
    out = out_dir / f"{edf_path.stem}.fif"
    raw.save(out, overwrite=True, verbose="ERROR")
    return out

def main(raw_dir="data/raw", interim_dir="data/interim"):
    raw_dir, interim_dir = map(pathlib.Path, (raw_dir, interim_dir))
    interim_dir.mkdir(parents=True, exist_ok=True)
    for f in raw_dir.glob("*.edf"):
        edf_to_fif(f, interim_dir)
        print(f"✔ {f.name}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="data/raw")
    p.add_argument("--interim_dir", default="data/interim")
    main(**vars(p.parse_args()))

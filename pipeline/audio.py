"""Stage 4: audio synthesis. Builds the dry signal from the sound timeline and renders it through the RIR bank."""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample_poly


from ctf_ltv_direct import ctf_ltv_direct
from scene_common import SAMPLE_FILES, SOUND_CLASSES, WALK_SPEED, az_el_dist, load_json, save_json

SAMPLES_DIR = "SAMPLES"
STFT_WIN = 512
STEP_LENGTH_M = 0.70
LABEL_HOP_S = 0.1


def find_sample_files(sound_class, samples_dir=SAMPLES_DIR):
    folder = os.path.join(samples_dir, sound_class)
    files = sorted(glob.glob(os.path.join(folder, "*.wav"))) if os.path.isdir(folder) else []
    for name in SAMPLE_FILES.get(sound_class, []):
        p = os.path.join(samples_dir, name)
        if os.path.exists(p):
            files.append(p)
    return files


def load_wav(path, fs):
    file_fs, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    elif x.dtype.kind == "u":
        x = (x.astype(np.float64) - 128) / 128
    else:
        x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if file_fs != fs:
        g = np.gcd(int(file_fs), int(fs))
        x = resample_poly(x, fs // g, file_fs // g)
    return x - np.mean(x)


def active_rms(x, fs):
    n = max(1, int(0.02 * fs))
    frames = x[:len(x) // n * n].reshape(-1, n)
    p = np.mean(frames ** 2, axis=1)
    if len(p) == 0 or p.max() <= 0:
        return 0.0
    return float(np.sqrt(np.mean(p[p > p.max() * 1e-4])))


def _fade(x, fs, ms=10.0):
    n = min(int(ms * 1e-3 * fs), len(x) // 2)
    if n > 0:
        w = np.sin(np.linspace(0, np.pi / 2, n)) ** 2
        x[:n] *= w
        x[-n:] *= w[::-1]
    return x


def slice_footsteps(x, fs):
    n = int(0.01 * fs)
    env = np.sqrt(fftconvolve(x ** 2, np.ones(n) / n, mode="same").clip(0))
    thr, min_gap = 0.25 * env.max(), int(0.2 * fs)
    onsets, i = [], 0
    while i < len(env):
        if env[i] > thr:
            onsets.append(max(0, i - int(0.015 * fs)))
            i += min_gap
        else:
            i += 1
    steps = []
    for k, a in enumerate(onsets):
        b = min(onsets[k + 1] if k + 1 < len(onsets) else len(x), a + int(0.45 * fs))
        if b - a > int(0.05 * fs):
            steps.append(_fade(x[a:b].copy(), fs, 5.0))
    return steps


def synth_event(sound_class, duration_s, fs, rng, samples_dir=SAMPLES_DIR):
    spec = SOUND_CLASSES[sound_class]
    n = int(round(duration_s * fs))
    out = np.zeros(n)
    files = find_sample_files(sound_class, samples_dir)
    if not files:
        return out, {"file": None, "mode": "missing", "note": f"no sample for '{sound_class}': event is silent"}
    path = files[rng.integers(len(files))]
    x = load_wav(path, fs)
    rms = active_rms(x, fs)
    if rms <= 0:
        return out, {"file": path, "mode": "missing", "note": "sample is silent"}
    x = x * (10 ** (spec["level_db"] / 20) / rms)
    mode = spec["mode"]
    info = {"file": path, "mode": mode}

    if mode == "steps":
        steps = slice_footsteps(x, fs)
        if len(steps) >= 2:
            interval = STEP_LENGTH_M / WALK_SPEED
            t, count = 0.5 * interval, 0
            while t < duration_s:
                s = steps[rng.integers(len(steps))] * 10 ** (rng.uniform(-1.5, 1.5) / 20)
                a = int(t * fs)
                b = min(n, a + len(s))
                out[a:b] += s[:b - a]
                t += interval * rng.uniform(0.95, 1.05)
                count += 1
            info.update(footsteps_in_sample=len(steps), footsteps_played=count,
                        cadence_hz=round(1 / interval, 2))
            return _fade(out, fs, 5.0), info
        info["mode"] = mode = "loop"
        info["note"] = "no single footsteps found in the sample, playing it as a loop"

    if mode == "oneshot":
        nz = np.nonzero(np.abs(x) > 0.02 * np.abs(x).max())[0]
        x = x[max(0, nz[0] - int(0.005 * fs)):]
        m = min(n, len(x))
        out[:m] = x[:m]
        return _fade(out, fs), info


    xf = int(0.03 * fs)
    if len(x) <= 2 * xf:
        x = np.tile(x, int(np.ceil(3 * xf / len(x))))
    start = int(rng.integers(len(x)))
    x = np.roll(x, -start)
    body = x[:-xf].copy()
    w = np.linspace(0, 1, xf)
    body[:xf] = body[:xf] * w + x[-xf:] * (1 - w)
    out[:] = np.tile(body, int(np.ceil(n / len(body))))[:n]
    info["loop_start_s"] = round(start / fs, 3)
    return _fade(out, fs), info


def build_dry_signal(timeline, duration_s, fs, rng, samples_dir=SAMPLES_DIR):
    dry = np.zeros(int(round(duration_s * fs)))
    infos = []
    for ev in timeline:
        x, info = synth_event(ev["sound_class"], ev["offset_s"] - ev["onset_s"], fs, rng, samples_dir)
        a = int(round(ev["onset_s"] * fs))
        m = min(len(x), len(dry) - a)
        dry[a:a + m] += x[:m]
        infos.append(info)
    return dry, infos


def prepare_bank(rirs, times, n_signal, fs, winsize=STFT_WIN):
    hop = winsize // 2
    L = rirs.shape[1]
    total = n_signal + L
    frames = np.round(np.asarray(times) * fs / hop).astype(int)
    frames[0] = 0
    keep = np.concatenate([[True], np.diff(frames) > 0])
    frames, irs = frames[keep], rirs[keep]
    end = max(int(np.ceil(total / hop)), frames[-1] + 1)
    frames, irs = np.append(frames, end), np.concatenate([irs, irs[-1:]], axis=0)
    return irs.T.astype(np.float64), frames * hop / fs, total


def convolve_scene(dry, rirs, times, fs, winsize=STFT_WIN, print_progress=False):
    irs, t, total = prepare_bank(rirs, times, len(dry), fs, winsize)
    sig = np.zeros((int(round(t[-1] * fs)) + 2 * winsize, 1))
    sig[:len(dry), 0] = dry


    # ctf_ltv_direct expects at least two channels; duplicate the mono RIRs
    irs2 = np.repeat(irs[:, None, :], 2, axis=1)
    y = ctf_ltv_direct(sig, irs2, t, fs, winsize, print_progress=print_progress)[:, 0]
    out = np.zeros(total)
    m = min(total, len(y))
    out[:m] = y[:m]
    return out


def source_position(bank, t):
    return np.array([np.interp(t, bank["times"], bank["pos_m"][:, k]) for k in range(3)])


def build_labels(timeline, bank, duration_s):
    rows = []
    for f in range(int(np.ceil(duration_s / LABEL_HOP_S))):
        t = (f + 0.5) * LABEL_HOP_S
        for ev in timeline:
            if ev["onset_s"] <= t < ev["offset_s"]:
                az, el, d = az_el_dist(source_position(bank, t), bank["mic_pos_m"])
                rows.append((f, ev["class_index"], 0, int(round(az)), int(round(el)), int(round(d * 100))))
    return rows


def render_scene(trajectory, bank, seed=0, samples_dir=SAMPLES_DIR, peak=0.9, print_progress=False):
    fs = int(bank["fs"])
    timeline = trajectory["sound_timeline"]
    rng = np.random.default_rng(seed)
    dry, infos = build_dry_signal(timeline, trajectory["duration_s"], fs, rng, samples_dir)
    wet = convolve_scene(dry, bank["rirs"], bank["times"], fs, print_progress=print_progress)
    gain = peak / max(np.max(np.abs(wet)), 1e-12)

    meta = []
    for ev, info in zip(timeline, infos):
        a, b = int(ev["onset_s"] * fs), int(ev["offset_s"] * fs)
        level = 20 * np.log10(np.sqrt(np.mean(wet[a:b] ** 2)) + 1e-12)
        meta.append({**ev, "starss23_class": SOUND_CLASSES[ev["sound_class"]]["starss23"],
                     "sample": info, "wet_level_db": round(float(level), 2)})
    info = {"fs": fs, "channels": 1, "duration_s": len(wet) / fs,
            "n_rirs": int(len(bank["times"])), "spacing_m": float(bank["spacing_m"]),
            "scene_gain_db": round(float(20 * np.log10(gain)), 2), "seed": seed,
            "mic_pos_m": np.asarray(bank["mic_pos_m"]).tolist(),
            "classes": {k: v["index"] for k, v in SOUND_CLASSES.items()},
            "label_format": "frame(100 ms), class, source, azimuth_deg, elevation_deg, distance_cm",
            "events": meta}
    return wet * gain, dry, build_labels(timeline, bank, trajectory["duration_s"]), info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 4: Audio Synthesiser")
    ap.add_argument("--trajectory", default="trajectory.json")
    ap.add_argument("--rirs", default=os.path.join("rirs", "rir_bank.npz"))
    ap.add_argument("--samples", default=SAMPLES_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="scene_audio.wav")
    args = ap.parse_args()

    traj = load_json(args.trajectory)
    bank = dict(np.load(args.rirs))
    print(f"Timeline: {len(traj['sound_timeline'])} sound events, {traj['duration_s']:.1f} s. "
          f"RIR bank: {len(bank['times'])} RIRs. Convolving ...")
    wet, dry, labels, info = render_scene(traj, bank, args.seed, args.samples)

    for e in info["events"]:
        s = e["sample"]
        print(f"  {e['onset_s']:6.2f} to {e['offset_s']:6.2f} s  {e['sound_class']:10s} {s['mode']:8s} "
              f"level at mic {e['wet_level_db']:6.1f} dB  "
              f"{os.path.basename(s['file']) if s['file'] else 'NO SAMPLE FOUND, EVENT IS SILENT'}")
    fs = info["fs"]
    wavfile.write(args.out, fs, np.round(wet * 32767).astype(np.int16))
    base = os.path.splitext(args.out)[0].replace("_audio", "")
    wavfile.write(base + "_dry.wav", fs,
                  np.round(dry / max(np.max(np.abs(dry)), 1e-12) * 0.9 * 32767).astype(np.int16))
    with open(base + "_labels.csv", "w") as f:
        f.writelines(",".join(str(v) for v in r) + "\n" for r in labels)
    save_json(info, base + "_events.json")
    print(f"\nSaved {args.out} ({info['duration_s']:.1f} s, {fs} Hz), {base}_dry.wav, "
          f"{base}_labels.csv ({len(labels)} rows), {base}_events.json")

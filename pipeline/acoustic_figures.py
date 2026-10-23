"""Figures describing the acoustic rendering: energy decay, RT60 per band, DRR and example RIRs."""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from pyroom import measure_rt60
from scene_common import OCTAVE_BANDS, load_json
from visualise_audio import BLUE, INK, MUTED, ORANGE, SURFACE

C_SOUND = 343.0


def edc_db(h):
    e = np.cumsum(h[::-1] ** 2)[::-1]
    return 10 * np.log10(e / e[0] + 1e-30)


def drr_db(h, dist, fs, window_ms=2.5):
    n0 = int(round(dist / C_SOUND * fs))
    w = int(window_ms * 1e-3 * fs)
    direct = np.sum(h[max(0, n0 - w):n0 + w + 1] ** 2)
    rest = np.sum(h ** 2) - direct
    return 10 * np.log10(direct / max(rest, 1e-30))


def load_bank(path):
    b = dict(np.load(path))
    fs = int(b["fs"])
    dist = np.linalg.norm(b["pos_m"] - b["mic_pos_m"], axis=1)
    return b["rirs"].astype(float), b["times"], dist, fs


def fig_edc(banks, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, which in zip(axes, ("nearest", "farthest")):
        for (label, rirs, times, dist, fs), style in zip(banks, ("-", "--")):
            i = int(np.argmin(dist) if which == "nearest" else np.argmax(dist))
            h = rirs[i]
            t = np.arange(len(h)) / fs
            rt = measure_rt60(h, fs)
            ax.plot(t, edc_db(h), style, color=BLUE if style == "-" else ORANGE, linewidth=1.8,
                    label=f"{label}: RT60 {rt:.2f} s" if np.isfinite(rt) else label)
        ax.set_title(f"{which} position, {dist[i]:.1f} m from the microphone", fontsize=9, loc="left", color=INK)
        ax.set_xlabel("Time (s)")
        ax.set_ylim(-70, 2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Energy decay (dB)")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_rt60_bands(banks, report, out):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    x = np.arange(len(OCTAVE_BANDS))
    ax.plot(x, report["sabine_s"], color=MUTED, linewidth=1.2, linestyle="--", label="Sabine")
    ax.plot(x, report["eyring_s"], color=MUTED, linewidth=1.2, label="Eyring")
    for (label, rirs, times, dist, fs), col in zip(banks, (BLUE, ORANGE)):
        h = rirs[int(np.argmax(dist))]
        meas = [measure_rt60(h, fs, b) for b in OCTAVE_BANDS]
        ax.plot(x, meas, color=col, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1, label=f"measured, {label}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}" if b < 1000 else f"{b // 1000}k" for b in OCTAVE_BANDS])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Octave band (Hz)")
    ax.set_ylabel("Reverberation time (s)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_drr(banks, out):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for (label, rirs, times, dist, fs), col in zip(banks, (BLUE, ORANGE)):
        d = np.array([drr_db(h, di, fs) for h, di in zip(rirs, dist)])
        order = np.argsort(dist)
        ax.plot(dist[order], d[order], linestyle="none", marker="o", markersize=4, color=col, alpha=0.6,
                markeredgewidth=0, label=label)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Source to microphone distance (m)")
    ax.set_ylabel("Direct-to-reverberant ratio (dB)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_examples(bank, out, show_ms=120):
    label, rirs, times, dist, fs = bank
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.4), sharex=True, sharey=True)
    for ax, which in zip(axes, ("nearest", "farthest")):
        i = int(np.argmin(dist) if which == "nearest" else np.argmax(dist))
        n = int(show_ms * 1e-3 * fs)
        t = np.arange(n) / fs * 1000
        ax.plot(t, rirs[i][:n], color=INK, linewidth=0.6)
        ax.axvline(dist[i] / C_SOUND * 1000, color=ORANGE, linewidth=1, linestyle="--")
        ax.set_title(f"{which} position, {dist[i]:.1f} m (dashed: direct-sound arrival)", fontsize=9,
                     loc="left", color=INK)
        ax.set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rirs", default=os.path.join("rirs", "rir_bank.npz"))
    ap.add_argument("--label", default="ISM")
    ap.add_argument("--compare", default=None, help="second RIR bank of the same trajectory")
    ap.add_argument("--compare-label", default="ISM + ray tracing")
    args = ap.parse_args()

    os.makedirs("figures", exist_ok=True)
    report = load_json(os.path.join(os.path.dirname(args.rirs), "room_acoustics.json"))
    banks = [(args.label, *load_bank(args.rirs))]
    if args.compare:
        banks.append((args.compare_label, *load_bank(args.compare)))
    fig_edc(banks, os.path.join("figures", "fig_edc.png"))
    fig_rt60_bands(banks, report, os.path.join("figures", "fig_rt60_bands.png"))
    fig_drr(banks, os.path.join("figures", "fig_drr_distance.png"))
    fig_examples(banks[-1], os.path.join("figures", "fig_rir_examples.png"))
    print("Saved figures/fig_edc.png, fig_rt60_bands.png, fig_drr_distance.png, fig_rir_examples.png")

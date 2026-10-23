"""Figures for a rendered scene: room and path, waveform, distance and spectrogram."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Rectangle
from scipy.io import wavfile
from scipy.signal import spectrogram

from scene_common import FURNITURE, corners_m, grid_point_m, load_json, tile_centre_m

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
CLASS_COLOURS = {"footsteps": BLUE, "water_tap": AQUA, "cooking": ORANGE, "cutlery": YELLOW,
                 "keyboard": VIOLET, "chair": MAGENTA, "tv": GREEN, "speech": RED}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
})


def _draw_room(ax, sm):
    g = sm["grid"]
    c = corners_m(sm).T
    ax.add_patch(Polygon(c, closed=True, facecolor="#f3f2ee", edgecolor=INK, linewidth=1.6, zorder=1))
    for key, meta in sm["furniture_metadata"].items():
        x0, y0 = grid_point_m(meta["top_left"][0], meta["top_left"][1] + meta["size"][1], g["scale_m"], g["height"])
        w, h = meta["size"][0] * g["scale_m"], meta["size"][1] * g["scale_m"]
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor="#d9d7cf", edgecolor=MUTED, linewidth=0.8, zorder=2))
        ax.text(x0 + w / 2, y0 + h / 2, FURNITURE[key]["label"].replace(" ", "\n"), ha="center", va="center",
                fontsize=7, color=MUTED, zorder=3)
    mx, my = tile_centre_m(*sm["mic_pos"], g["scale_m"], g["height"])
    ax.plot(mx, my, marker="^", color=INK, markersize=9, linestyle="none", zorder=6, label="Microphone")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.grid(False)
    ax.set_xlim(c[:, 0].min() - 0.4, c[:, 0].max() + 0.4)
    ax.set_ylim(c[:, 1].min() - 0.4, c[:, 1].max() + 0.4)


def fig_room(sm, traj, out):
    g = sm["grid"]
    centre = lambda p: tile_centre_m(p[0], p[1], g["scale_m"], g["height"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

    ax = axes[0]
    _draw_room(ax, sm)
    raw = []
    for s in traj["raw_trajectory"]:
        try:
            raw.append(centre((float(s["position"][0]), float(s["position"][1]))))
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    if raw:
        r = np.array(raw)
        ax.plot(r[:, 0], r[:, 1], color=ORANGE, linewidth=1.4, marker="o", markersize=4, zorder=5,
                label="Proposed path (before repair)", clip_on=False)
    n_v = sum(traj["raw_violations"].values())
    ax.set_ylabel("y (m)")
    ax.text(0.0, 1.03, f"(a) proposed by the planner: {len(raw)} steps, {n_v} rule violations",
            transform=ax.transAxes, fontsize=9, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)

    ax = axes[1]
    _draw_room(ax, sm)
    p = np.array([centre(s["position"]) for s in traj["trajectory"]])
    ax.plot(p[:, 0], p[:, 1], color=BLUE, linewidth=2, zorder=5, label="Validated path (after repair)")
    ax.plot(*p[0], marker="o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
            linestyle="none", zorder=7)
    ax.annotate("start", p[0], textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK, zorder=8)
    marks = []
    for st in traj["trajectory"]:
        if st["action"] not in ("walking", "standing") and (not marks or marks[-1][1] != tuple(st["position"])):
            marks.append((st["action"], tuple(st["position"])))
    for k, (act, tile) in enumerate(marks):
        x, y = centre(tile)
        ax.plot(x, y, marker="o", color=SURFACE, markersize=13, markeredgecolor=BLUE, markeredgewidth=2,
                linestyle="none", zorder=7)
        ax.text(x, y, str(k + 1), ha="center", va="center", fontsize=8, color=INK, zorder=8)
    if marks:
        ax.text(0.0, -0.3, "   ".join(f"{k + 1}: {a.replace('_', ' ')}" for k, (a, _) in enumerate(marks)),
                transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.text(0.0, 1.03, f"(b) after validation: {len(traj['trajectory'])} steps, {traj['duration_s']:.1f} s",
            transform=ax.transAxes, fontsize=9, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_audio(traj, bank, out, wav_path="scene_audio.wav"):
    fs, x = wavfile.read(wav_path)
    x = x.astype(float) / 32768
    t = np.arange(len(x)) / fs
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, height_ratios=[1.2, 0.8, 1.6])

    ax = axes[0]
    ax.plot(t, x, color=INK, linewidth=0.4)
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.3, 1.05)
    seen = set()
    for e in traj["sound_timeline"]:
        col = CLASS_COLOURS[e["sound_class"]]
        for a in axes[:2]:
            a.axvspan(e["onset_s"], e["offset_s"], color=col, alpha=0.13, linewidth=0)
        ax.plot([e["onset_s"], e["offset_s"]], [-1.15, -1.15], color=col, linewidth=4, solid_capstyle="butt",
                label=None if e["sound_class"] in seen else e["sound_class"].replace("_", " "))
        seen.add(e["sound_class"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=max(len(seen), 1), fontsize=8)

    ax = axes[1]
    d = np.linalg.norm(bank["pos_m"] - bank["mic_pos_m"], axis=1)
    ax.plot(bank["times"], d, color=INK, linewidth=1.6)
    ax.set_ylabel("Source to\nmicrophone (m)")
    ax.set_ylim(bottom=0)

    ax = axes[2]
    f, tt, S = spectrogram(x, fs, nperseg=1024, noverlap=768)
    SdB = 10 * np.log10(S + 1e-14)
    im = ax.pcolormesh(tt, f / 1000, SdB, cmap=SEQ, vmin=SdB.max() - 70, vmax=SdB.max(), shading="auto",
                       rasterized=True)
    ax.set_ylabel("Frequency (kHz)")
    ax.set_xlabel("Time (s)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=axes, pad=0.01, fraction=0.025, aspect=40)
    cb.set_label("Power spectral density (dB)")
    cb.outline.set_visible(False)
    axes[0].set_xlim(0, t[-1])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _read_csv(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig_study(rows, out):
    rows = [r for r in rows if not r["error"]]
    measures = [("fully_valid", 100, "Fully valid answers (%)"),
                ("pct_positions_walkable", 1, "Positions on walkable tiles (%)"),
                ("pct_moves_one_tile", 1, "Moves of exactly one tile (%)"),
                ("pct_orientation_ok", 1, "Correct orientation (%)"),
                ("order_correct", 100, "Correct action order (%)"),
                ("repairs_total", 1, "Repairs needed per trajectory")]
    scen = sorted({r["scenario"] for r in rows})
    variants = [("1", "with walkable tile list", BLUE), ("0", "map only", ORANGE)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, (key, scale, label) in zip(axes.ravel(), measures):
        for v, name, col in variants:
            ys = []
            for sc in scen:
                vals = [float(r[key]) * scale for r in rows
                        if r["scenario"] == sc and r["tile_list_in_prompt"] == v and r[key] != ""]
                ys.append(np.mean(vals) if vals else np.nan)
            ax.plot(range(len(scen)), ys, color=col, linewidth=2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        ax.set_title(label, fontsize=9, loc="left", color=INK)
        ax.set_xticks(range(len(scen)))
        ax.set_xticklabels([s.split()[0] for s in scen])
        if "%" in label:
            ax.set_ylim(0, 105)
        else:
            ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("Furniture interactions in the scenario")
    axes[0, 1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_resolution(rows, out):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    sp = [float(r["spacing_m"]) for r in rows]
    for key, name, col in (("below_1k", "below 1 kHz", BLUE), ("all", "all frequencies", INK),
                           ("above_4k", "above 4 kHz", ORANGE)):
        ax.plot(sp, [float(r[f"level_error_mean_db_{key}"]) for r in rows], color=col, linewidth=2, marker="o",
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks(sp)
    ax.set_xticklabels([f"{v:g}" for v in sp])
    ax.minorticks_off()
    ax.invert_xaxis()
    ax.set_xlabel("Distance between RIR coordinates (m)")
    ax.set_ylabel("Mean level error while walking (dB)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_prompt(rows, measures, out, reference=None):
    rows = [r for r in rows if not r["error"]]
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    fig, axes = plt.subplots(1, len(measures), figsize=(3.1 * len(measures), 0.45 * len(conds) + 1.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (key, scale, label) in zip(np.atleast_1d(axes), measures):
        for i, c in enumerate(conds):
            vals = [float(r[key]) * scale for r in rows if r["condition"] == c and r.get(key) not in ("", None)]
            if not vals:
                continue
            ax.plot(vals, i + rng.uniform(-0.12, 0.12, len(vals)), linestyle="none", marker="o", markersize=4,
                    color=BLUE, alpha=0.35, markeredgewidth=0)
            ax.plot(np.mean(vals), i, linestyle="none", marker="o", markersize=10, color=BLUE,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        if reference and reference.get(key) is not None:
            ax.axvline(reference[key] * scale, color=ORANGE, linewidth=1.5)
        ax.set_title(label, fontsize=9, loc="left", color=INK)
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels(conds)
        ax.grid(axis="y", visible=False)
    np.atleast_1d(axes)[0].invert_yaxis()
    note = "Large dot: mean. Small dots: single runs."
    if reference:
        note += " Orange line: mean of the real floor plans."
    fig.text(0.01, -0.02, note, fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    made = []
    if os.path.exists("spatial_map.json") and os.path.exists("trajectory.json"):
        sm, traj = load_json("spatial_map.json"), load_json("trajectory.json")
        fig_room(sm, traj, os.path.join("figures", "fig_room_trajectory.png"))
        made.append("fig_room_trajectory.png")
    if made and os.path.exists("scene_audio.wav") and os.path.exists(os.path.join("rirs", "rir_bank.npz")):
        fig_audio(traj, dict(np.load(os.path.join("rirs", "rir_bank.npz"))),
                  os.path.join("figures", "fig_scene_audio.png"))
        made.append("fig_scene_audio.png")
    if os.path.exists(os.path.join("study", "trajectories.csv")):
        fig_study(_read_csv(os.path.join("study", "trajectories.csv")), os.path.join("figures", "fig_study.png"))
        made.append("fig_study.png")
    if os.path.exists(os.path.join("study", "resolution.csv")):
        fig_resolution(_read_csv(os.path.join("study", "resolution.csv")),
                       os.path.join("figures", "fig_resolution.png"))
        made.append("fig_resolution.png")
    if os.path.exists(os.path.join("prompt_study", "layout.csv")):
        import main
        from prompt_study import REF_DIR, _realism
        real = [_realism(main.layout_metrics(sm)) for _, sm in main.load_references(REF_DIR)]
        ref = {k: np.mean([r[k] for r in real if r[k] is not None]) for k in real[0]}
        fig_prompt(_read_csv(os.path.join("prompt_study", "layout.csv")),
                   [("furniture_valid_pct", 1, "Furniture placed validly (%)"),
                    ("wall_items_pct", 1, "Wall items at a wall (%)"),
                    ("tv_sofa_distance_m", 1, "TV to sofa (m)"),
                    ("bed_kitchen_distance_m", 1, "Bed to kitchen (m)"),
                    ("table_kitchen_distance_m", 1, "Table to kitchen (m), held out")],
                   os.path.join("figures", "fig_prompt_layout.png"), ref)
        made.append("fig_prompt_layout.png")
    if os.path.exists(os.path.join("prompt_study", "trajectory.csv")):
        fig_prompt(_read_csv(os.path.join("prompt_study", "trajectory.csv")),
                   [("pct_positions_walkable", 1, "Positions on walkable tiles (%)"),
                    ("pct_moves_one_tile", 1, "Moves of exactly one tile (%)"),
                    ("pct_orientation_ok", 1, "Correct orientation (%)"),
                    ("repairs_total", 1, "Repairs needed")],
                   os.path.join("figures", "fig_prompt_trajectory.png"))
        made.append("fig_prompt_trajectory.png")
    print("Saved figures/" + ", figures/".join(made))

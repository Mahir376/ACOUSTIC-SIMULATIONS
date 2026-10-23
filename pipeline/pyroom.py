"""Stage 3: acoustic rendering. Computes one room impulse response per trajectory coordinate with pyroomacoustics."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pyroomacoustics as pra
from scipy.signal import butter, sosfiltfilt

from scene_common import (
    FS, FURNITURE, MIC_HEIGHT, OCTAVE_BANDS, SOURCE_HEIGHT, corners_m, load_json,
    material_coeffs, save_json, tile_centre_m,
)

WALL_SCATTERING = 0.10
FLOOR_SCATTERING = 0.20
DEFAULT_SPACING_M = 0.25
ISM_ORDER = 6
ISM_ORDER_HYBRID = 3


def _material(coeffs, scattering, name):
    return pra.Material({"coeffs": [float(c) for c in coeffs], "center_freqs": OCTAVE_BANDS,
                         "description": name}, scattering)


def effective_floor_absorption(spatial_map):
    scale = spatial_map["grid"]["scale_m"]
    s_floor = spatial_map["floor_area_m2"]
    a_floor = material_coeffs(spatial_map["surfaces"]["floor"])
    s_hidden, a_furn, detail = 0.0, np.zeros(len(OCTAVE_BANDS)), {}
    for key, meta in spatial_map.get("furniture_metadata", {}).items():
        w, h = meta["size"][0] * scale, meta["size"][1] * scale
        footprint = w * h
        name = FURNITURE[key]["material"]
        s = footprint if name.startswith("chairs_") else footprint + 2 * (w + h) * FURNITURE[key]["height"]
        a = material_coeffs(name) * s
        s_hidden += footprint
        a_furn += a
        detail[key] = {"material": name, "area_m2": round(s, 2),
                       "absorption_area_m2": [round(float(v), 3) for v in a]}
    eff = (a_floor * (s_floor - s_hidden) + a_furn) / s_floor
    return np.clip(eff, 0.01, 0.99), detail


def build_room(spatial_map, fs=FS, ray_tracing=False, ism_order=None, n_rays=20000):
    corners = corners_m(spatial_map)
    surf = spatial_map["surfaces"]
    wall_abs = material_coeffs(surf["walls"])
    ceil_abs = material_coeffs(surf["ceiling"])
    floor_abs, _ = effective_floor_absorption(spatial_map)
    if spatial_map.get("_test_uniform_alpha") is not None:
        wall_abs = ceil_abs = floor_abs = np.full(len(OCTAVE_BANDS), float(spatial_map["_test_uniform_alpha"]))
    if ism_order is None:
        ism_order = ISM_ORDER_HYBRID if ray_tracing else ISM_ORDER
    room = pra.Room.from_corners(
        corners, fs=fs, max_order=ism_order, ray_tracing=ray_tracing, air_absorption=True,
        materials=[_material(wall_abs, WALL_SCATTERING, surf["walls"])] * corners.shape[1],
    )
    room.extrude(spatial_map["grid"]["ceiling_m"], materials={
        "floor": _material(floor_abs, FLOOR_SCATTERING, surf["floor"] + "+furniture"),
        "ceiling": _material(ceil_abs, WALL_SCATTERING, surf["ceiling"]),
    })
    if ray_tracing:
        room.set_ray_tracing(n_rays=n_rays, receiver_radius=0.3, hist_bin_size=0.004)
    return room


def mic_position(spatial_map):
    g = spatial_map["grid"]
    x, y = tile_centre_m(*spatial_map["mic_pos"], g["scale_m"], g["height"])
    return np.array([x, y, MIC_HEIGHT])


def interpolate_trajectory(spatial_map, steps, spacing_m=DEFAULT_SPACING_M):
    g = spatial_map["grid"]
    centre = lambda p: np.array(tile_centre_m(p[0], p[1], g["scale_m"], g["height"]))
    n_sub = max(1, int(round(g["scale_m"] / spacing_m)))
    times, pos = [steps[0]["t_start"]], [centre(steps[0]["position"])]
    for i, s in enumerate(steps):
        t0, t1 = s["t_start"], s["t_start"] + s["duration_s"]
        b = centre(s["position"])
        if s["action"] == "walking" and i > 0:
            a = centre(steps[i - 1]["position"])
            for k in range(1, n_sub + 1):
                times.append(t0 + (t1 - t0) * k / n_sub)
                pos.append(a + (b - a) * k / n_sub)
        else:
            times.append(t1)
            pos.append(b)
    pos = np.column_stack([np.array(pos), np.full(len(pos), SOURCE_HEIGHT)])
    return np.array(times), pos


def compute_rirs(spatial_map, positions, fs=FS, ray_tracing=False, ism_order=None, n_rays=20000, seed=0):
    pra.random.seed(seed)
    room = build_room(spatial_map, fs, ray_tracing, ism_order, n_rays)
    keys = [tuple(np.round(p, 4)) for p in positions]
    unique = list(dict.fromkeys(keys))
    for p in unique:
        room.add_source(list(p))
    room.add_microphone(mic_position(spatial_map))
    room.compute_rir()

    # pyroomacoustics delays every RIR by half the fractional delay filter length
    fdl2 = pra.constants.get("frac_delay_length") // 2
    rirs = [np.asarray(room.rir[0][s][fdl2:], dtype=np.float64) for s in range(len(unique))]
    L = max(len(r) for r in rirs)
    bank = np.stack([np.pad(r, (0, L - len(r))) for r in rirs])
    return bank[[unique.index(k) for k in keys]], room


def trim_tail(bank, floor_db=-65.0, fs=FS):
    e = np.max(bank ** 2, axis=0)
    edc = np.cumsum(e[::-1])[::-1]
    idx = np.nonzero(10 * np.log10(edc / edc[0] + 1e-30) > floor_db)[0]
    L = min(bank.shape[1], (idx[-1] if len(idx) else bank.shape[1] - 1) + int(0.02 * fs))
    out = bank[:, :L].astype(np.float32)
    fade = int(0.005 * fs)
    out[:, -fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return out


def predicted_rt60(room):
    V = room.get_volume()
    areas = np.array([w.area() for w in room.walls])
    alpha = np.array([np.atleast_1d(w.absorption) * np.ones(len(OCTAVE_BANDS)) for w in room.walls])
    S = areas.sum()
    A = (areas[:, None] * alpha).sum(axis=0)
    m4V = 4 * np.asarray(room.air_absorption) * V if room.air_absorption is not None else 0.0
    return {"volume_m3": float(V), "surface_m2": float(S), "mean_alpha": (A / S).tolist(),
            "sabine_s": (0.161 * V / (A + m4V)).tolist(),
            "eyring_s": (0.161 * V / (-S * np.log(1 - A / S) + m4V)).tolist()}


def measure_rt60(rir, fs, band_hz=None, lo_db=-5.0, hi_db=-25.0):
    h = np.asarray(rir, float)
    if band_hz:
        sos = butter(4, [band_hz / np.sqrt(2), min(band_hz * np.sqrt(2), 0.49 * fs)],
                     btype="band", fs=fs, output="sos")
        h = sosfiltfilt(sos, h)
    edc = np.cumsum(h[::-1] ** 2)[::-1]
    edc_db = 10 * np.log10(edc / edc[0] + 1e-30)
    i0, i1 = np.argmax(edc_db <= lo_db), np.argmax(edc_db <= hi_db)
    if i1 <= i0 + 10:
        return float("nan")
    return float(-60.0 / np.polyfit(np.arange(i0, i1) / fs, edc_db[i0:i1], 1)[0])


def compute_rir_bank(spatial_map, trajectory, spacing_m=DEFAULT_SPACING_M, fs=FS,
                     ray_tracing=False, n_rays=20000, seed=0, verbose=True):

    tile = spatial_map["grid"]["scale_m"]
    asked, spacing_m = spacing_m, tile / max(1, int(round(tile / spacing_m)))
    if verbose and abs(asked - spacing_m) > 1e-9:
        print(f"Note: {asked} m does not divide a {tile} m tile evenly, using {spacing_m} m instead.")
    times, pos = interpolate_trajectory(spatial_map, trajectory["trajectory"], spacing_m)
    mic = mic_position(spatial_map)
    n_unique = len({tuple(np.round(p, 4)) for p in pos})
    if verbose:
        print(f"Trajectory: {len(trajectory['trajectory'])} steps -> {len(pos)} RIR coordinates "
              f"at {spacing_m} m spacing ({n_unique} distinct positions)")
        print(f"Simulating with {'ISM + ray tracing' if ray_tracing else 'ISM'} ...")
    t = time.time()
    rirs, room = compute_rirs(spatial_map, pos, fs, ray_tracing, None, n_rays, seed)
    rirs = trim_tail(rirs, fs=fs)
    if verbose:
        print(f"  done in {time.time() - t:.1f} s, RIR length {rirs.shape[1] / fs * 1000:.0f} ms")

    dist = np.linalg.norm(pos - mic, axis=1)
    peak = np.max(np.abs(rirs), axis=1)
    floor_eff, furn_detail = effective_floor_absorption(spatial_map)
    far = int(np.argmax(dist))
    report = {
        "fs": fs, "simulator": "ism+ray_tracing" if ray_tracing else "ism",
        "ism_order": ISM_ORDER_HYBRID if ray_tracing else ISM_ORDER,
        "n_rays": n_rays if ray_tracing else None, "spacing_m": spacing_m,
        "n_rirs": int(len(pos)), "n_distinct_positions": int(n_unique),
        "mic_pos_m": mic.tolist(), "octave_bands_hz": OCTAVE_BANDS,
        "surfaces": spatial_map["surfaces"],
        "absorption": {"walls": material_coeffs(spatial_map["surfaces"]["walls"]).tolist(),
                       "ceiling": material_coeffs(spatial_map["surfaces"]["ceiling"]).tolist(),
                       "floor_bare": material_coeffs(spatial_map["surfaces"]["floor"]).tolist(),
                       "floor_with_furniture": floor_eff.tolist()},
        "furniture_absorption": furn_detail,
        **predicted_rt60(room),
        "measured_rt60_s": [measure_rt60(rirs[far], fs, b) for b in OCTAVE_BANDS],
        "measured_rt60_note": ("measured on the simulated RIR farthest from the microphone" if ray_tracing else
                               "ISM only: reflections stop at the maximum order, so the measured value "
                               "is shorter than the room's real reverberation time"),
        "rir_length_s": rirs.shape[1] / fs,
        "distance_to_mic_m": [round(float(dist.min()), 3), round(float(dist.max()), 3)],
        "per_rir": [{"t_s": round(float(a), 3), "pos_m": [round(float(v), 3) for v in p],
                     "distance_m": round(float(d), 3), "peak": round(float(k), 4)}
                    for a, p, d, k in zip(times, pos, dist, peak)],
    }
    bank = {"rirs": rirs, "times": times, "pos_m": pos, "mic_pos_m": mic, "fs": fs, "spacing_m": spacing_m}
    return bank, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 3: Acoustic Renderer")
    ap.add_argument("--map", default="spatial_map.json")
    ap.add_argument("--trajectory", default="trajectory.json")
    ap.add_argument("--out-dir", default="rirs")
    ap.add_argument("--spacing", type=float, default=DEFAULT_SPACING_M,
                    help="distance between RIR coordinates in metres (0.5 = one per tile)")
    ap.add_argument("--ray-tracing", action="store_true", help="hybrid ISM + ray tracing")
    ap.add_argument("--rays", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sm, traj = load_json(args.map), load_json(args.trajectory)
    bank, report = compute_rir_bank(sm, traj, args.spacing, ray_tracing=args.ray_tracing,
                                    n_rays=args.rays, seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, "rir_bank.npz"), **bank)
    save_json(report, os.path.join(args.out_dir, "room_acoustics.json"))

    f3 = lambda v: " ".join(f"{x:5.2f}" for x in v)
    print(f"\nRoom: {report['volume_m3']:.1f} m^3, surface {report['surface_m2']:.1f} m^2, "
          f"source {report['distance_to_mic_m'][0]} to {report['distance_to_mic_m'][1]} m from the microphone")
    print("band (Hz)         " + " ".join(f"{b:5d}" for b in OCTAVE_BANDS))
    print("RT60 Sabine   (s) " + f3(report["sabine_s"]))
    print("RT60 Eyring   (s) " + f3(report["eyring_s"]))
    print("RT60 measured (s) " + f3(report["measured_rt60_s"]) + f"   ({report['measured_rt60_note']})")
    print(f"\nSaved {args.out_dir}/rir_bank.npz ({report['n_rirs']} RIRs) and {args.out_dir}/room_acoustics.json")

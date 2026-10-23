"""Automatic checks of the acoustic stages against independent references."""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

import action
import audio
import pyroom
from main import _clearance, _floor_connected
from scene_common import (
    MIC_HEIGHT, access_tiles, az_el_dist, build_grid, load_json, polygon_area, save_json, tile_centre_m,
)

RESULTS = []


def check(stage, name, ok, value, expect):
    RESULTS.append({"stage": stage, "check": name, "pass": bool(ok), "measured": value, "expected": expect})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {value}   (expected {expect})")


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-30))


def tv_convolve_exact(x, irs, times, fs):
    N, L = irs.shape
    y = np.zeros(len(x) + L - 1)
    idx = np.clip(np.round(np.asarray(times) * fs).astype(int), 0, len(x))
    for i in range(N):
        a = 0 if i == 0 else idx[i - 1]
        b = len(x) if i == N - 1 else idx[i + 1]
        if b <= a:
            continue
        w = np.ones(b - a)
        if i > 0 and idx[i] > a:
            w[:idx[i] - a] = np.linspace(0, 1, idx[i] - a, endpoint=False)
        if i < N - 1 and b > idx[i]:
            w[idx[i] - a:] = np.linspace(1, 0, b - idx[i], endpoint=False)
        y[a:b + L - 1] += fftconvolve(x[a:b] * w, irs[i])
    return y


def stage1(sm):
    print("\nSTAGE 1  Spatial Architect")
    g, scale = sm["grid"], sm["grid"]["scale_m"]
    grid = build_grid(sm)
    corners = sm["room_corners"]
    rect = all(a[0] == b[0] or a[1] == b[1] for a, b in zip(corners, corners[1:] + corners[:1]))
    check(1, "polygon is rectilinear", rect, f"{len(corners)} vertices", "all edges axis-aligned")

    base = build_grid({**sm, "furniture_metadata": {}, "mic_pos": None}, with_mic=False)
    n_floor = sum(r.count("_") for r in base)
    area_poly = abs(polygon_area(corners)) * scale ** 2
    check(1, "tile floor area equals polygon area", abs(n_floor * scale ** 2 - area_poly) < 1e-6,
          f"{n_floor * scale ** 2:.2f} m^2 vs {area_poly:.2f} m^2", "identical")
    leak = sum(1 for y, r in enumerate(base) for x, c in enumerate(r) if c == "_"
               and any(not (0 <= y + dy < g["height"] and 0 <= x + dx < g["width"]) or base[y + dy][x + dx] == " "
                       for dx in (-1, 0, 1) for dy in (-1, 0, 1)))
    check(1, "walls enclose the floor", leak == 0, f"{leak} floor tiles touching outside", "0")

    room = pyroom.build_room(sm)
    pt = lambda x, y: np.array([*tile_centre_m(x, y, scale, g["height"]), 1.0])
    inside = [room.is_inside(pt(x, y)) for y, r in enumerate(base) for x, c in enumerate(r) if c == "_"]
    outside = [room.is_inside(pt(x, y)) for y, r in enumerate(base) for x, c in enumerate(r)
               if c != "_" and 0 < x < g["width"] - 1 and 0 < y < g["height"] - 1]
    check(1, "tile map and acoustic room describe the same space", all(inside) and not any(outside),
          f"{sum(inside)}/{len(inside)} floor tiles inside, {sum(outside)} other tiles inside", "all / 0")
    check(1, "acoustic volume equals floor area x height",
          abs(room.get_volume() - area_poly * g["ceiling_m"]) < 0.01 * room.get_volume(),
          f"{room.get_volume():.2f} m^3 vs {area_poly * g['ceiling_m']:.2f} m^3", "within 1 %")

    cells, overlap, off = set(), 0, 0
    for meta in sm["furniture_metadata"].values():
        for dy in range(meta["size"][1]):
            for dx in range(meta["size"][0]):
                c = (meta["top_left"][0] + dx, meta["top_left"][1] + dy)
                overlap += c in cells
                off += base[c[1]][c[0]] != "_"
                cells.add(c)
    check(1, "furniture on the floor, no overlap", overlap == 0 and off == 0,
          f"{off} tiles off the floor, {overlap} overlapping", "0, 0")
    reach = {k: len(access_tiles(grid, sm, k)) for k in sm["furniture_metadata"]}
    check(1, "every furniture item is reachable",
          min(reach.values()) > 0 and _floor_connected([[("_" if c == "_" else "#") for c in r] for r in grid]),
          f"free neighbouring tiles {reach}", "> 0 each, floor connected")
    mx, my = sm["mic_pos"]
    g2 = [r[:] for r in grid]
    g2[my][mx] = "_"
    check(1, "microphone on a free floor tile", grid[my][mx] == "M" and base[my][mx] == "_",
          f"tile {sm['mic_pos']}, {_clearance(g2, mx, my) * scale:.1f} m of free space around it", "free floor tile")
    return grid, room


def stage2(sm, traj, grid):
    print("\nSTAGE 2  Action Scripter")
    steps, scale = traj["trajectory"], sm["grid"]["scale_m"]
    try:
        action.assert_valid(steps, grid, sm)
        ok, msg = True, f"{len(steps)} steps valid"
    except AssertionError as e:
        ok, msg = False, str(e)
    check(2, "final path: walkable, one tile per move, interactions next to their furniture", ok, msg, "no violation")
    t_ok = all(abs(steps[i]["t_start"] + steps[i]["duration_s"] - steps[i + 1]["t_start"]) < 1e-3
               for i in range(len(steps) - 1))
    check(2, "timeline is continuous", t_ok, f"total {traj['duration_s']:.2f} s", "no gaps or overlaps")
    walk = [s for s in steps if s["action"] == "walking"]
    speeds = [scale / s["duration_s"] for s in walk] or [0]
    check(2, "walking speed", not walk or (max(speeds) - min(speeds) < 1e-3 and 0.8 <= speeds[0] <= 1.8),
          f"{speeds[0]:.2f} m/s over {len(walk)} tiles", "0.8 to 1.8 m/s, constant")
    revisit, run = 0, []
    for s in steps:
        if s["action"] == "walking":
            revisit += tuple(s["position"]) in run
            run.append(tuple(s["position"]))
        else:
            run = [tuple(s["position"])]
    check(2, "no back-and-forth loops while walking", revisit == 0, f"{revisit} revisited tiles", "0")
    tl = traj["sound_timeline"]
    tl_ok = all(a["offset_s"] <= b["onset_s"] + 1e-6 for a, b in zip(tl[:-1], tl[1:])) and all(
        0 <= e["onset_s"] < e["offset_s"] <= traj["duration_s"] + 1e-6 for e in tl)
    check(2, "sound timeline is ordered and inside the scene", tl_ok, f"{len(tl)} events", "no overlap, all within the scene")
    print(f"  [INFO] LLM answer before repair: {len(traj['raw_trajectory'])} steps, "
          f"violations {traj['raw_violations']}")
    print(f"  [INFO] repairs applied: { {k: v for k, v in traj['repair_report'].items() if v} or 'none'}")


def stage3(sm, traj, bank, rep, room):
    print("\nSTAGE 3  Acoustic Renderer")
    fs = int(bank["fs"])
    mic, pos, times = bank["mic_pos_m"], bank["pos_m"], bank["times"]
    c = room.c

    check(3, "all RIR coordinates inside the room", all(room.is_inside(p) for p in pos), f"{len(pos)} coordinates", "all inside")
    dmin = float(np.min(np.linalg.norm(pos - mic, axis=1)))
    check(3, "source never coincides with the microphone", dmin >= 0.4, f"closest {dmin:.2f} m", ">= 0.4 m")
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    check(3, "spacing between neighbouring RIR coordinates", step.max() <= float(bank["spacing_m"]) + 1e-6,
          f"max {step.max():.3f} m", f"<= {float(bank['spacing_m'])} m")
    check(3, "RIR time stamps increase", np.all(np.diff(times) >= 0) and abs(times[-1] - traj["duration_s"]) < 1e-3,
          f"0 to {times[-1]:.2f} s", f"0 to {traj['duration_s']:.2f} s")


    g, scale = sm["grid"], sm["grid"]["scale_m"]
    base = build_grid({**sm, "furniture_metadata": {}, "mic_pos": None}, with_mic=False)
    pts = []
    for y, r in enumerate(base):
        for x, ch in enumerate(r):
            if ch == "_" and _clearance(base, x, y) >= 2:
                p = np.array([*tile_centre_m(x, y, scale, g["height"]), MIC_HEIGHT])
                d = np.linalg.norm(p - mic)
                if 0.7 <= d <= 4.0 and all(room.is_inside(mic + (p - mic) * a) for a in np.linspace(0.02, 0.98, 60)):
                    pts.append(p)
    pts = pts[::max(1, len(pts) // 40)]
    out = {"dist_m": [], "direct_amp": []}
    if pts:
        rirs, _ = pyroom.compute_rirs(sm, np.array(pts), fs, ism_order=2)
        delay_err, amp_err = [], []
        for h, p in zip(rirs, pts):
            d = np.linalg.norm(p - mic)
            n = int(round(d / c * fs))
            w = h[max(0, n - 6):n + 7]
            delay_err.append(max(0, n - 6) + np.argmax(np.abs(w)) - d / c * fs)
            amp_err.append(db(np.sum(w ** 2)) - db(1 / d ** 2))
            out["dist_m"].append(float(d))
            out["direct_amp"].append(float(np.sqrt(np.sum(w ** 2))))
        check(3, "direct-sound delay equals distance / c", np.max(np.abs(delay_err)) <= 0.6,
              f"max error {np.max(np.abs(delay_err)):.2f} samples over {len(pts)} positions", "<= 0.6 samples")
        check(3, "direct-sound level follows 1/r", np.max(np.abs(amp_err)) <= 0.5,
              f"max deviation {np.max(np.abs(amp_err)):.2f} dB (0.7 to 4 m)", "<= 0.5 dB")


        far = max(pts, key=lambda p: np.linalg.norm(p - mic))
        h_u, room_u = pyroom.compute_rirs({**sm, "_test_uniform_alpha": 0.25}, np.array([far]), fs, ray_tracing=True)
        meas = [pyroom.measure_rt60(h_u[0], fs, b) for b in (500, 1000, 2000)]
        eyr = [pyroom.predicted_rt60(room_u)["eyring_s"][i] for i in (2, 3, 4)]
        dev = float(np.max(np.abs(np.array(meas) / np.array(eyr) - 1)) * 100)
        check(3, "hybrid simulator, uniform absorption 0.25, against Eyring's formula", dev <= 25,
              f"measured {np.round(meas, 2).tolist()} s, Eyring {np.round(eyr, 2).tolist()} s, max {dev:.0f} %",
              "within 25 % (500 Hz to 2 kHz)")
    if rep["simulator"] != "ism":
        m, e = np.array(rep["measured_rt60_s"]), np.array(rep["eyring_s"])
        print(f"  [INFO] scene reverberation time: measured {np.round(m, 2).tolist()} s, "
              f"Eyring {np.round(e, 2).tolist()} s (Eyring assumes evenly spread absorption)")
    else:
        print(f"  [INFO] ISM only (order {rep['ism_order']}): RIRs are {rep['rir_length_s'] * 1000:.0f} ms long; "
              f"Eyring predicts a reverberation time of {rep['eyring_s'][3]:.2f} s at 1 kHz for this room. "
              f"Use --ray-tracing for the full decay.")
    return out


def stage4(sm, traj, bank, room):
    print("\nSTAGE 4  Audio Synthesiser")
    fs = int(bank["fs"])
    rirs, times, mic = bank["rirs"].astype(float), bank["times"], bank["mic_pos_m"]
    rng = np.random.default_rng(1)
    x = rng.standard_normal(fs)


    h = rirs[len(rirs) // 2]
    ref = fftconvolve(x, h)
    y = audio.convolve_scene(x, np.stack([h, h, h]), np.array([0.0, 0.3, 0.8]), fs)
    n = min(len(y), len(ref))
    err = db(np.sum((y[:n] - ref[:n]) ** 2) / np.sum(ref[:n] ** 2))
    check(4, "ctf_ltv_direct, stationary source, against exact convolution", err < -60, f"error {err:.0f} dB", "< -60 dB")


    moving = np.nonzero(np.linalg.norm(np.diff(bank["pos_m"], axis=0), axis=1) > 0)[0]
    if len(moving) > 4:
        a = moving[0]
        b = min(len(times) - 1, a + 40)
        T = times[a:b + 1] - times[a]
        xs = x[:min(len(x), int(T[-1] * fs))]
        irs_p, t_p, _ = audio.prepare_bank(rirs[a:b + 1], T, len(xs), fs)
        yc = audio.convolve_scene(xs, rirs[a:b + 1], T, fs)
        ye = tv_convolve_exact(xs, irs_p.T, t_p, fs)
        n = min(len(yc), len(ye))
        lag = int(np.argmax(fftconvolve(yc[:n], ye[:n][::-1])) - (n - 1))
        lvl = db(np.sum(yc[:n] ** 2) / np.sum(ye[:n] ** 2))
        check(4, "ctf_ltv_direct, moving source, against time-domain reference", lag == 0 and abs(lvl) < 1.0,
              f"lag {lag} samples, level difference {lvl:+.2f} dB", "lag 0, level within 1 dB")

    wfs, wav = wavfile.read("scene_audio.wav")
    wav = wav.astype(float) / 32768
    check(4, "output format", wfs == fs and wav.ndim == 1 and np.max(np.abs(wav)) < 0.999,
          f"{wfs} Hz, mono, peak {np.max(np.abs(wav)):.2f}, {len(wav) / fs:.1f} s", f"{fs} Hz, mono, no clipping")


    _, dry = wavfile.read("scene_dry.wav")
    dry = dry.astype(float) / 32768
    active = np.zeros(len(dry), dtype=bool)
    for e in traj["sound_timeline"]:
        active[int(e["onset_s"] * fs):int(e["offset_s"] * fs) + 1] = True
    leak = float(np.max(np.abs(dry[~active]))) if (~active).any() else 0.0
    check(4, "dry signal is silent outside the timeline events", leak < 1e-3, f"peak {leak:.4f}", "< 0.001")

    lab = np.loadtxt("scene_labels.csv", delimiter=",", ndmin=2) if os.path.getsize("scene_labels.csv") else np.zeros((0, 6))
    bad = 0
    for f, cls, _, az, el, dist in lab:
        t = (f + 0.5) * audio.LABEL_HOP_S
        act = [e for e in traj["sound_timeline"] if e["onset_s"] <= t < e["offset_s"] and e["class_index"] == cls]
        if not act:
            bad += 1
            continue
        a2, e2, d2 = az_el_dist(audio.source_position(bank, t), mic)
        bad += abs(((a2 - az + 180) % 360) - 180) > 1 or abs(e2 - el) > 1 or abs(d2 * 100 - dist) > 1
    expected = sum(1 for f in range(int(np.ceil(traj["duration_s"] / audio.LABEL_HOP_S)))
                   for e in traj["sound_timeline"] if e["onset_s"] <= (f + 0.5) * audio.LABEL_HOP_S < e["offset_s"])
    check(4, "labels match the timeline and the source positions", bad == 0 and len(lab) == expected,
          f"{len(lab)} rows, {bad} mismatches, {expected} expected", "0 mismatches, counts equal")
    info = load_json("scene_events.json")
    missing = [m["sound_class"] for m in info["events"] if m["sample"]["mode"] == "missing"]
    check(4, "every timeline event has a sample", not missing, f"missing: {missing or 'none'}", "none missing")
    if os.path.exists(os.path.join(audio.SAMPLES_DIR, "SOURCES.md")):
        print("  [INFO] SAMPLES/ holds the synthetic source signals (see SAMPLES/SOURCES.md).")


if __name__ == "__main__":
    sm, traj = load_json("spatial_map.json"), load_json("trajectory.json")
    bank = dict(np.load(os.path.join("rirs", "rir_bank.npz")))
    rep = load_json(os.path.join("rirs", "room_acoustics.json"))

    grid, room = stage1(sm)
    stage2(sm, traj, grid)
    free_field = stage3(sm, traj, bank, rep, room)
    stage4(sm, traj, bank, room)

    n_fail = sum(not r["pass"] for r in RESULTS)
    print(f"\n{len(RESULTS) - n_fail} of {len(RESULTS)} checks passed.")
    save_json({"checks": RESULTS, "free_field": free_field}, "validation_report.json")
    print("Saved validation_report.json")
    sys.exit(1 if n_fail else 0)

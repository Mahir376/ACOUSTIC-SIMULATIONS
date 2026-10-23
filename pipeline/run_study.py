"""Capability study: generates layouts and trajectories over shapes, scenarios and prompts and summarises the audit results."""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

import action
import main
from scene_common import build_grid, load_json, save_json

STUDY_DIR = "study"


SCENARIOS = {
    "2 interactions": (
        "The actor walks to the kitchen counter and runs the tap to get a glass of water, "
        "then walks to the table and types on a laptop.",
        [["KK"], ["TTTT"]]),
    "3 interactions": (
        "The actor cooks a meal at the kitchen counter, walks to the table to eat it, "
        "then goes to the sofa to watch TV.",
        [["KK"], ["TTTT"], ["SSSS", "TV"]]),
    "4 interactions": (
        "The actor runs the tap at the kitchen counter, eats at the table, watches TV from the sofa, "
        "and finally goes to the bed to rest.",
        [["KK"], ["TTTT"], ["SSSS", "TV"], ["BB"]]),
}


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(path, row, fields):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


LAYOUT_FIELDS = ["layout_id", "shape", "layout_prompt", "run", "source", "error",
                 "polygon_valid_first_try", "polygon_attempts", "polygon_valid_final",
                 "furniture_valid", "furniture_total", "mic_valid", "agent_start_valid",
                 "floor_area_m2", "n_corners", "wall_items_at_wall", "tv_sofa_distance_m",
                 "bed_kitchen_distance_m"]
TRAJ_FIELDS = ["layout_id", "shape", "layout_prompt", "layout_source", "scenario", "tile_list_in_prompt", "error",
               "raw_steps", "violations_total", "fully_valid", "pct_positions_walkable",
               "pct_moves_one_tile", "pct_interactions_ok", "pct_orientation_ok", "pct_sound_ok",
               "moving_non_walk", "starts_at_start", "order_correct", "targets_in_order",
               "repairs_total", "final_steps", "path_efficiency", "duration_s"]


def order_correct(visited, expected):
    i = 0
    for t in visited:
        if i < len(expected) and t in expected[i]:
            i += 1
        elif i > 0 and t in expected[i - 1]:
            continue
        else:
            return False
    return i == len(expected)


PROMPT_TAGS = {"examples": "ex", "no-examples": "noex", "references": "ref"}


def _check_header(path, fields):
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            if next(csv.reader(f), None) != fields:
                raise SystemExit(f"{path} was written by an older version of this script. "
                                 f"Rename or move the '{STUDY_DIR}' folder and start again.")


def run_capability_study(shapes, runs, layout_prompts, given_map=None, references_dir=None):
    os.makedirs(os.path.join(STUDY_DIR, "runs"), exist_ok=True)
    lay_csv, traj_csv = os.path.join(STUDY_DIR, "layouts.csv"), os.path.join(STUDY_DIR, "trajectories.csv")
    _check_header(lay_csv, LAYOUT_FIELDS)
    _check_header(traj_csv, TRAJ_FIELDS)
    references = main.load_references(references_dir) if "references" in layout_prompts else None
    done_lay = {r["layout_id"] for r in read_rows(lay_csv) if not r["error"]}
    done_traj = {(r["layout_id"], r["scenario"], r["tile_list_in_prompt"])
                 for r in read_rows(traj_csv) if not r["error"]}

    jobs = [(shape, ex, run) for shape in shapes for ex in layout_prompts for run in range(1, runs + 1)]
    n_calls = len(jobs) * (1 + 2 * len(SCENARIOS)) + (runs * 2 * len(SCENARIOS) if given_map else 0)
    print(f"Capability study: {len(jobs)} layouts, about {n_calls} LLM calls in total "
          f"(a rejected room polygon costs up to two more).\n")

    def trajectories(layout_id, shape, sm, layout_prompt):
        grid = build_grid(sm)
        for name, (text, expected) in SCENARIOS.items():
            for tiles in (True, False):
                if (layout_id, name, str(int(tiles))) in done_traj:
                    continue
                row = {"layout_id": layout_id, "shape": shape, "layout_prompt": layout_prompt,
                       "layout_source": sm["source"],
                       "scenario": name, "tile_list_in_prompt": int(tiles), "error": ""}
                print(f"[{layout_id}] {name}, tile list {'on' if tiles else 'off'}")
                try:

                    res = action.run(sm, text, with_tile_list=tiles, max_attempts=1)
                    m = res["metrics"]
                    row.update(m)
                    row["targets_in_order"] = ">".join(m["targets_in_order"])
                    row["order_correct"] = int(order_correct(m["targets_in_order"], expected))
                    row["repairs_total"] = sum(res["repair_report"].values())
                    row["duration_s"] = res["duration_s"]
                    tag = name.split()[0] + ("_tiles" if tiles else "_notiles")
                    save_json(res, os.path.join(STUDY_DIR, "runs", f"{layout_id}_traj_{tag}.json"))
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"[:200]
                    print("    failed:", row["error"])
                append_row(traj_csv, row, TRAJ_FIELDS)

    for shape, ex, run in jobs:
        layout_id = f"{shape.split('-')[0]}_{PROMPT_TAGS[ex]}_{run:02d}"
        map_path = os.path.join(STUDY_DIR, "runs", f"{layout_id}_map.json")
        if layout_id not in done_lay:
            print(f"[{layout_id}] requesting a {shape} layout")
            row = {"layout_id": layout_id, "shape": shape, "layout_prompt": ex, "run": run,
                   "source": "llm", "error": ""}
            try:
                scenario = f"A {shape} studio flat with a kitchen corner, a living area and a bed."
                sm, log = main.generate_spatial_map(scenario, with_examples=(ex != "no-examples"),
                                                    references=references if ex == "references" else None)
                row["polygon_valid_first_try"] = int(not log[0]["problems"])
                row["polygon_attempts"] = len(log)
                row["polygon_valid_final"] = int(sm is not None)
                if sm is not None:
                    r = sm["raw_layout"]
                    row.update({"furniture_valid": r["furniture_valid"], "furniture_total": r["furniture_total"],
                                "mic_valid": int(r["mic_valid"]), "agent_start_valid": int(r["agent_start_valid"]),
                                **sm["layout_metrics"]})
                    save_json(sm, map_path)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"[:200]
                print("    failed:", row["error"])
            append_row(lay_csv, row, LAYOUT_FIELDS)
        if os.path.exists(map_path):
            trajectories(layout_id, shape, load_json(map_path), ex)

    if given_map:
        sm = main.load_given_map(given_map)
        save_json(sm, os.path.join(STUDY_DIR, "runs", "given_map.json"))
        if "given_01" not in done_lay:
            append_row(lay_csv, {"layout_id": "given_01", "shape": "given", "layout_prompt": "given",
                                 "source": "given", "run": 1,
                                 "error": "", **sm["layout_metrics"]}, LAYOUT_FIELDS)
        for run in range(1, runs + 1):
            trajectories(f"given_{run:02d}", "given", sm, "given")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean(rows, key):
    vals = [_num(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def _table(title, rows, group_keys, columns):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)
    head = [k.replace("_", " ") for k in group_keys] + ["n"] + [c[0] for c in columns]
    body = [[*g, str(len(rs))] + [f"{_mean(rs, key) * scale:.{dec}f}" for _, key, scale, dec in columns]
            for g, rs in sorted(groups.items())]
    widths = [max(len(str(x)) for x in col) for col in zip(head, *body)] if body else [len(h) for h in head]
    line = lambda cells: "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))
    return "\n".join([title, line(head), line(["-" * w for w in widths])] + [line(b) for b in body]) + "\n"


def summarise(references_dir=os.path.join("maps", "real")):
    lay = [r for r in read_rows(os.path.join(STUDY_DIR, "layouts.csv")) if not r["error"] and r["source"] == "llm"]
    traj = [r for r in read_rows(os.path.join(STUDY_DIR, "trajectories.csv")) if not r["error"]]
    for r in lay:
        ft = _num(r.get("furniture_total"))
        r["furniture_valid_pct"] = 100 * _num(r["furniture_valid"]) / ft if ft else ""
    out = []
    lay_cols = [("wall items at wall %", "wall_items_pct", 1, 0),
                ("TV to sofa m", "tv_sofa_distance_m", 1, 1),
                ("bed to kitchen m", "bed_kitchen_distance_m", 1, 1),
                ("floor area m2", "floor_area_m2", 1, 1)]
    all_lay = [r for r in read_rows(os.path.join(STUDY_DIR, "layouts.csv"))
               if not r["error"] and r["source"] == "llm"]
    try:
        for name, sm in main.load_references(references_dir):
            all_lay.append({"layout_prompt": "(real floor plans)", **{k: str(v) if v is not None else ""
                            for k, v in main.layout_metrics(sm).items()}})
    except FileNotFoundError:
        pass
    for r in all_lay:
        try:
            a, b = r["wall_items_at_wall"].split("/")
            r["wall_items_pct"] = 100 * int(a) / int(b)
        except (ValueError, ZeroDivisionError, AttributeError):
            r["wall_items_pct"] = ""
    if lay:
        out.append(_table("LAYOUT REALISM by prompt (wall items = counter, bed, sofa, TV)",
                          [r for r in all_lay if r.get("floor_area_m2")], ["layout_prompt"], lay_cols))
        out.append(_table("LAYOUTS: what the LLM got right without correction", lay,
                          ["shape", "layout_prompt"],
                          [("polygon ok 1st try %", "polygon_valid_first_try", 100, 0),
                           ("polygon ok after retries %", "polygon_valid_final", 100, 0),
                           ("furniture valid %", "furniture_valid_pct", 1, 0),
                           ("mic valid %", "mic_valid", 100, 0),
                           ("start valid %", "agent_start_valid", 100, 0),
                           ("floor area m2", "floor_area_m2", 1, 1),
                           ("TV to sofa m", "tv_sofa_distance_m", 1, 1)]))
    cols = [("fully valid %", "fully_valid", 100, 0),
            ("walkable %", "pct_positions_walkable", 1, 1),
            ("one-tile moves %", "pct_moves_one_tile", 1, 1),
            ("interactions ok %", "pct_interactions_ok", 1, 1),
            ("orientation ok %", "pct_orientation_ok", 1, 1),
            ("sound ok %", "pct_sound_ok", 1, 1),
            ("order correct %", "order_correct", 100, 0),
            ("repairs", "repairs_total", 1, 1),
            ("path efficiency", "path_efficiency", 1, 2)]
    if traj:
        out.append(_table("TRAJECTORIES by scenario length and prompt (first LLM answer, before repair)",
                          traj, ["scenario", "tile_list_in_prompt"], cols))
        out.append(_table("TRAJECTORIES by room shape", traj, ["shape"], cols))
    text = "\n".join(out) if out else "No results yet.\n"
    print("\n" + text)
    os.makedirs(STUDY_DIR, exist_ok=True)
    with open(os.path.join(STUDY_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)


def resolution_test(spacings=(0.5, 0.25, 0.125, 0.0625, 0.03125), ray_tracing=False):
    from scipy.io import wavfile
    from scipy.signal import butter, sosfiltfilt

    import audio
    import pyroom

    sm, traj = load_json("spatial_map.json"), load_json("trajectory.json")
    fs = pyroom.FS
    sigma = 0.05
    dry = sigma * np.random.default_rng(0).standard_normal(int(traj["duration_s"] * fs))
    walking = np.zeros(len(dry), dtype=bool)
    for s in traj["trajectory"]:
        if s["action"] == "walking":
            walking[int(s["t_start"] * fs):int((s["t_start"] + s["duration_s"]) * fs)] = True

    bands = {"all": None, "below_1k": ("low", 1000), "above_4k": ("high", 4000)}
    filt = lambda x, b: x if b is None else sosfiltfilt(butter(6, b[1], btype=b[0], fs=fs, output="sos"), x, axis=-1)

    n = int(0.05 * fs)
    n_frames = len(dry) // n
    t_frame = (np.arange(n_frames) + 0.5) * n / fs
    walk_frames = walking[:n_frames * n].reshape(n_frames, n).all(axis=1)

    print("Resolution test: computing the true level curve from RIRs every 0.03125 m ...")
    fine, _ = pyroom.compute_rir_bank(sm, traj, 0.03125, ray_tracing=ray_tracing, verbose=False)
    truth = {}
    for name, b in bands.items():
        e = np.sum(filt(fine["rirs"].astype(float), b) ** 2, axis=1)
        truth[name] = 10 * np.log10(sigma ** 2 * np.interp(t_frame, fine["times"], e))

    os.makedirs(STUDY_DIR, exist_ok=True)
    rows, peak = [], None
    for sp in spacings:
        bank = fine if sp == 0.03125 else pyroom.compute_rir_bank(sm, traj, sp, ray_tracing=ray_tracing, verbose=False)[0]
        y = audio.convolve_scene(dry, bank["rirs"], bank["times"], fs)
        row = {"spacing_m": sp, "n_rirs": len(bank["times"]),
               "delay_step_ms": round(sp / 343.0 * 1000, 3)}
        for name, b in bands.items():
            yb = filt(y[:n_frames * n], b).reshape(n_frames, n)
            err = (10 * np.log10(np.mean(yb ** 2, axis=1) + 1e-20) - truth[name])[walk_frames]
            row[f"level_error_mean_db_{name}"] = round(float(err.mean()), 2)
            row[f"level_error_worst_db_{name}"] = round(float(np.percentile(np.abs(err), 95)), 2)
        rows.append(row)
        peak = peak or np.max(np.abs(y))
        wavfile.write(os.path.join(STUDY_DIR, f"resolution_{sp}m.wav"), fs,
                      np.round(y / peak * 0.5 * 32767).clip(-32767, 32767).astype(np.int16))
        print(f"  {sp} m done ({row['n_rirs']} RIRs)")
    path = os.path.join(STUDY_DIR, "resolution.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nRESOLUTION: level of a moving white-noise source minus the true level, while walking")
    print("            mean error (dB), and in brackets the error that 95 % of frames stay below")
    print("spacing m   RIRs   all frequencies    below 1 kHz        above 4 kHz")
    for r in rows:
        cell = lambda k: f"{r['level_error_mean_db_' + k]:+.2f} ({r['level_error_worst_db_' + k]:.2f})".ljust(17)
        print(f"{r['spacing_m']:<10}  {r['n_rirs']:<5}  {cell('all')}  {cell('below_1k')}  {cell('above_4k')}")
    print(f"\nSaved {path} and study/resolution_<spacing>m.wav (listen to them side by side)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Capability study and resolution test")
    ap.add_argument("--runs", type=int, default=10, help="layouts per room shape")
    ap.add_argument("--shapes", nargs="+", default=main.SHAPES, choices=main.SHAPES)
    ap.add_argument("--layout-prompts", choices=["examples", "no-examples", "references", "both", "all"],
                    default="examples", help="both = examples + no-examples; all = those two + references")
    ap.add_argument("--references-dir", default=os.path.join("maps", "real"),
                    help="folder of traced real floor plans used by the 'references' prompt")
    ap.add_argument("--given-map", default=None, help="hand-drawn tile map to add as a condition")
    ap.add_argument("--summary", action="store_true", help="only print the tables from existing results")
    ap.add_argument("--resolution", action="store_true", help="run the resolution test instead")
    ap.add_argument("--ray-tracing", action="store_true", help="resolution test with the hybrid simulator")
    args = ap.parse_args()

    if args.resolution:
        resolution_test(ray_tracing=args.ray_tracing)
    else:
        if not args.summary:
            prompts = {"examples": ["examples"], "no-examples": ["no-examples"], "references": ["references"],
                       "both": ["examples", "no-examples"],
                       "all": ["examples", "no-examples", "references"]}[args.layout_prompts]
            run_capability_study(args.shapes, args.runs, prompts, args.given_map, args.references_dir)
        summarise(args.references_dir)

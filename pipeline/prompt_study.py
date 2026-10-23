"""Prompt experiment: compares room prompt variants (baseline, rules, references) on validity and realism measures."""

from __future__ import annotations

import argparse
import glob
import os
import statistics

import action
import main
from run_study import SCENARIOS, append_row, order_correct, read_rows
from scene_common import build_grid, load_json, save_json

OUT = "prompt_study"
REF_DIR = os.path.join("maps", "real")

LAYOUT_CONDITIONS = {
    "baseline":   (False, False, False),
    "rules":      (True,  False, False),
    "references": (False, True,  False),
    "reasoning":  (False, False, True),
    "all":        (True,  True,  True),
}
TRAJ_CONDITIONS = {
    "baseline":   (True,  False, False),
    "map-only":   (False, False, False),
    "plan-first": (True,  True,  False),
    "self-check": (True,  False, True),
    "all":        (True,  True,  True),
}

LAYOUT_FIELDS = ["condition", "shape", "run", "error", "polygon_valid_first_try", "polygon_attempts",
                 "polygon_valid_final", "furniture_valid_pct", "mic_valid", "agent_start_valid",
                 "wall_items_pct", "tv_sofa_distance_m", "tv_sofa_2_to_4_m", "bed_kitchen_distance_m",
                 "table_kitchen_distance_m", "floor_area_m2"]
TRAJ_FIELDS = ["condition", "room", "scenario", "run", "error", "raw_steps", "violations_total", "fully_valid",
               "pct_positions_walkable", "pct_moves_one_tile", "pct_interactions_ok", "pct_orientation_ok",
               "pct_sound_ok", "order_correct", "repairs_total", "path_efficiency"]


def _realism(metrics):
    out = {k: metrics.get(k) for k in ("tv_sofa_distance_m", "tv_sofa_2_to_4_m", "bed_kitchen_distance_m",
                                       "table_kitchen_distance_m", "floor_area_m2")}
    a, b = metrics["wall_items_at_wall"].split("/")
    out["wall_items_pct"] = round(100 * int(a) / int(b), 1) if int(b) else None
    return out


def run_layout(conditions, shapes, runs):
    os.makedirs(os.path.join(OUT, "runs"), exist_ok=True)
    path = os.path.join(OUT, "layout.csv")
    done = {(r["condition"], r["shape"], r["run"]) for r in read_rows(path) if not r["error"]}
    refs = main.load_references(REF_DIR)
    todo = [(c, s, r) for r in range(1, runs + 1) for s in shapes for c in conditions
            if (c, s, str(r)) not in done]
    print(f"Layout prompt study: {len(todo)} layouts to request "
          f"({len(conditions)} conditions x {len(shapes)} shape(s) x {runs} runs, minus those already done).\n")
    for n, (cond, shape, run) in enumerate(todo, 1):
        rules, use_refs, reasoning = LAYOUT_CONDITIONS[cond]
        print(f"[{n}/{len(todo)}] {cond}, {shape}, run {run}")
        row = {"condition": cond, "shape": shape, "run": run, "error": ""}
        try:
            scenario = f"A {shape} studio flat with a kitchen corner, a living area and a bed."
            sm, log = main.generate_spatial_map(scenario, references=refs if use_refs else None,
                                                rules=rules, reasoning=reasoning)
            row.update(polygon_valid_first_try=int(not log[0]["problems"]), polygon_attempts=len(log),
                       polygon_valid_final=int(sm is not None))
            if sm is not None:
                r = sm["raw_layout"]
                row.update(furniture_valid_pct=round(100 * r["furniture_valid"] / r["furniture_total"], 1),
                           mic_valid=int(r["mic_valid"]), agent_start_valid=int(r["agent_start_valid"]),
                           **_realism(sm["layout_metrics"]))
                save_json(sm, os.path.join(OUT, "runs", f"layout_{cond}_{shape.split('-')[0]}_{run:02d}.json"))
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"[:200]
            print("    failed:", row["error"])
        append_row(path, row, LAYOUT_FIELDS)


def run_trajectory(conditions, scenario_names, runs):
    os.makedirs(os.path.join(OUT, "runs"), exist_ok=True)
    path = os.path.join(OUT, "trajectory.csv")
    done = {(r["condition"], r["room"], r["scenario"], r["run"]) for r in read_rows(path) if not r["error"]}
    rooms = {os.path.splitext(os.path.basename(f))[0]: main.load_given_map(f)
             for f in sorted(glob.glob(os.path.join(REF_DIR, "*.txt")))}

    usable = {name: [r for r, sm in rooms.items()
                     if all(any(k in sm["furniture_metadata"] for k in group) for group in SCENARIOS[name][1])]
              for name in scenario_names}
    todo = [(c, room, sc, r) for r in range(1, runs + 1) for sc in scenario_names for room in usable[sc]
            for c in conditions if (c, room, sc, str(r)) not in done]
    print(f"Trajectory prompt study: {len(todo)} trajectories to request "
          f"({len(conditions)} conditions, rooms per scenario "
          f"{ {k: len(v) for k, v in usable.items()} }, {runs} runs).\n")
    for n, (cond, room, sc, run) in enumerate(todo, 1):
        tiles, plan_first, self_check = TRAJ_CONDITIONS[cond]
        text, expected = SCENARIOS[sc]
        print(f"[{n}/{len(todo)}] {cond}, {room}, {sc}, run {run}")
        row = {"condition": cond, "room": room, "scenario": sc, "run": run, "error": ""}
        try:

            res = action.run(rooms[room], text, with_tile_list=tiles, max_attempts=1,
                             plan_first=plan_first, self_check=self_check)
            m = res["metrics"]
            row.update({k: m[k] for k in TRAJ_FIELDS if k in m})
            row["order_correct"] = int(order_correct(m["targets_in_order"], expected))
            row["repairs_total"] = sum(res["repair_report"].values())
            save_json(res, os.path.join(OUT, "runs", f"traj_{cond}_{room[:12]}_{sc.split()[0]}_{run:02d}.json"))
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"[:200]
            print("    failed:", row["error"])
        append_row(path, row, TRAJ_FIELDS)


def _stat(rows, key, dec=1):
    vals = []
    for r in rows:
        try:
            vals.append(float(r.get(key)))
        except (TypeError, ValueError):
            pass
    if not vals:
        return "-"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{statistics.mean(vals):.{dec}f} ({sd:.{dec}f})"


def _table(title, groups, columns):
    head = ["condition", "n"] + [c[0] for c in columns]
    body = []
    for label, rows in groups:
        cells = []
        for _, key, dec, scale in columns:
            scaled = [{key: (float(r[key]) * scale if r.get(key) not in ("", None) else "")} for r in rows]
            cells.append(_stat(scaled, key, dec))
        body.append([label, str(len(rows))] + cells)
    widths = [max(len(str(x)) for x in col) for col in zip(head, *body)]
    line = lambda cells: "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))
    return "\n".join([title, line(head), line(["-" * w for w in widths])] + [line(b) for b in body]) + "\n"


def summarise():
    out = []
    lay = [r for r in read_rows(os.path.join(OUT, "layout.csv")) if not r["error"]]
    if lay:
        groups = [(c, [r for r in lay if r["condition"] == c]) for c in LAYOUT_CONDITIONS]
        groups = [g for g in groups if g[1]]
        out.append(_table("LAYOUT PROMPT, validity: mean (standard deviation)", groups, [
            ("polygon ok 1st try %", "polygon_valid_first_try", 0, 100),
            ("furniture valid %", "furniture_valid_pct", 0, 1),
            ("mic valid %", "mic_valid", 0, 100),
            ("start valid %", "agent_start_valid", 0, 100)]))
        real = [{k: ("" if v is None else str(v)) for k, v in _realism(main.layout_metrics(sm)).items()}
                for _, sm in main.load_references(REF_DIR)]
        out.append(_table("LAYOUT PROMPT, realism: mean (standard deviation). Last column is the held-out measure.",
                          groups + [("(real floor plans)", real)], [
            ("wall items at wall %", "wall_items_pct", 0, 1),
            ("TV to sofa m", "tv_sofa_distance_m", 1, 1),
            ("TV-sofa 2 to 4 m %", "tv_sofa_2_to_4_m", 0, 100),
            ("bed to kitchen m", "bed_kitchen_distance_m", 1, 1),
            ("table to kitchen m", "table_kitchen_distance_m", 1, 1)]))
    traj = [r for r in read_rows(os.path.join(OUT, "trajectory.csv")) if not r["error"]]
    if traj:
        groups = [(c, [r for r in traj if r["condition"] == c]) for c in TRAJ_CONDITIONS]
        out.append(_table("TRAJECTORY PROMPT, first answer before repair: mean (standard deviation)",
                          [g for g in groups if g[1]], [
            ("fully valid %", "fully_valid", 0, 100),
            ("walkable %", "pct_positions_walkable", 1, 1),
            ("one-tile moves %", "pct_moves_one_tile", 1, 1),
            ("interactions ok %", "pct_interactions_ok", 1, 1),
            ("orientation ok %", "pct_orientation_ok", 1, 1),
            ("order correct %", "order_correct", 0, 100),
            ("repairs", "repairs_total", 1, 1),
            ("path efficiency", "path_efficiency", 2, 1)]))
    failed = sum(1 for f in ("layout.csv", "trajectory.csv") for r in read_rows(os.path.join(OUT, f)) if r["error"])
    text = "\n".join(out) if out else "No results yet.\n"
    if failed:
        text += f"\n{failed} request(s) failed and are not in the tables (see the 'error' column in the CSV files).\n"
    print("\n" + text)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)


def write_prompts():
    folder = os.path.join(OUT, "prompts")
    os.makedirs(folder, exist_ok=True)
    refs = main.load_references(REF_DIR)
    for name, (rules, use_refs, reasoning) in LAYOUT_CONDITIONS.items():
        with open(os.path.join(folder, f"layout_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(main.system_prompt(True, refs if use_refs else None, rules, reasoning))
    room = main.load_given_map(sorted(glob.glob(os.path.join(REF_DIR, "*.txt")))[0])
    grid = build_grid(room)
    for name, (tiles, plan_first, self_check) in TRAJ_CONDITIONS.items():
        with open(os.path.join(folder, f"trajectory_{name}.txt"), "w", encoding="utf-8") as f:
            f.write("=== SYSTEM PROMPT ===\n" + action.system_prompt(plan_first, self_check) +
                    "\n\n=== USER PROMPT (example room) ===\n" +
                    action.build_user_prompt(room, grid, SCENARIOS["3 interactions"][0], tiles))
    print(f"Wrote {len(LAYOUT_CONDITIONS) + len(TRAJ_CONDITIONS)} prompt files to {folder}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prompt engineering study")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("layout", help="compare layout prompts")
    p1.add_argument("--runs", type=int, default=10)
    p1.add_argument("--shapes", nargs="+", default=["T-shaped"], choices=main.SHAPES)
    p1.add_argument("--conditions", nargs="+", default=list(LAYOUT_CONDITIONS), choices=list(LAYOUT_CONDITIONS))
    p2 = sub.add_parser("trajectory", help="compare trajectory prompts")
    p2.add_argument("--runs", type=int, default=5)
    p2.add_argument("--scenarios", nargs="+", default=["3 interactions"], choices=list(SCENARIOS))
    p2.add_argument("--conditions", nargs="+", default=list(TRAJ_CONDITIONS), choices=list(TRAJ_CONDITIONS))
    sub.add_parser("summary", help="print the tables from existing results")
    sub.add_parser("prompts", help="write the exact prompt texts to files")
    args = ap.parse_args()

    if args.cmd == "layout":
        run_layout(args.conditions, args.shapes, args.runs)
    elif args.cmd == "trajectory":
        run_trajectory(args.conditions, args.scenarios, args.runs)
    elif args.cmd == "prompts":
        write_prompts()
    if args.cmd != "prompts":
        summarise()

"""Stage 2: script generation. Reads spatial_map.json, asks the LLM for the actor's script and writes trajectory.json."""

from __future__ import annotations

import argparse

from scene_common import (
    ACTIONS, FURNITURE, SOUND_CLASSES, WALK_SPEED,
    access_tiles, ascii_with_axes, astar, build_grid, chat_json, is_walkable,
    load_json, nearest_walkable, normalise_action, orientation_between, save_json,
    walkable_tiles,
)

DEFAULT_SCENARIO = (
    "The actor walks to the kitchen counter and runs the tap to get a glass of water, "
    "then walks to the table and types on a laptop."
)
ORIENTATIONS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}

_BASE_PROMPT = """You plan the movement of one person (the actor) inside a small flat drawn on a tile grid.

MAP LEGEND
  _ floor (walkable)      W wall        M microphone on a stand (not walkable)
  K kitchen counter (KK)  S sofa (SSSS) T table (TTTT)  B bed (BB)  V television (TV)
Coordinates are [x, y]: x is the column (left to right), y is the row (top to bottom).
Orientation: north = towards smaller y, south = larger y, east = larger x, west = smaller x.

RULES
1. The first step is at the actor start tile with action "standing".
2. Every position must be a walkable floor tile ('_').
3. Consecutive positions are either identical or exactly one tile apart
   (up, down, left or right; no diagonals, no jumps).
4. A step whose position changes must have the action "walking", and its
   orientation must be the direction of that move.
5. To use a furniture item the actor first walks to a floor tile directly next to it,
   then performs the action while staying on that tile, facing the item, with
   "target" set to the item.
6. Actions must follow a sensible cause-and-effect order for the scenario.
7. Allowed actions: walking, standing, running_water, cooking, eating, typing,
   sitting_down, watching_tv, speaking, resting.
8. Give "duration_s" (1 to 10 seconds) for every action other than walking.
9. For every step give "sound": the sound you expect to hear from that action, one of
   footsteps, water_tap, cooking, cutlery, keyboard, chair, tv, speech, none.
10. Use 15 to 45 steps.

%(extra)sReturn ONLY this JSON object:
{
  "narrative": "one sentence describing what the actor does and why",%(plan_field)s
  "trajectory": [
    {"step": 1, "position": [x, y], "orientation": "north", "action": "standing",
     "target": null, "duration_s": 1.5, "sound": "none", "description": "short description"}
  ]
}"""


PLAN_FIRST = """PLAN FIRST
Before writing the steps, fill the "plan" field: for every furniture item the scenario uses, in
order, give the item, the floor tile next to it where the actor will stand, and the action there.
Then write the trajectory so that it visits exactly these tiles in this order."""

SELF_CHECK = """CHECK BEFORE ANSWERING
Go through your trajectory step by step and verify: the position is a walkable floor tile; it is
identical to or exactly one tile from the previous position; the orientation matches the move or
faces the furniture in use. Correct every step that fails before you answer."""


def system_prompt(plan_first=False, self_check=False) -> str:
    extra = (PLAN_FIRST + "\n\n" if plan_first else "") + (SELF_CHECK + "\n\n" if self_check else "")
    plan_field = ('\n  "plan": [{"target": "KK", "stand_at": [x, y], "action": "running_water"}],'
                  if plan_first else "")
    return _BASE_PROMPT % {"extra": extra, "plan_field": plan_field}


SYSTEM_PROMPT = system_prompt()


def build_user_prompt(spatial_map, grid, scenario, with_tile_list=True, feedback=""):
    furn = [f"- {key}: {FURNITURE[key]['label']}" for key in spatial_map["furniture_metadata"]]
    p = (f"SCENARIO: {scenario}\n\nMAP:\n{ascii_with_axes(grid)}\n\n"
         f"ACTOR START: {spatial_map['agent_start']}\n\nFURNITURE:\n" + "\n".join(furn))
    if with_tile_list:
        p += ("\n\nWALKABLE TILES (every position must come from this list):\n"
              + ", ".join(str(list(t)) for t in walkable_tiles(grid)))
        p += "\n\nTILES NEXT TO EACH FURNITURE ITEM:\n" + "\n".join(
            f"- {k}: {[list(t) for t in access_tiles(grid, spatial_map, k)]}"
            for k in spatial_map["furniture_metadata"])
    return p + feedback


def generate_trajectory(spatial_map, grid, scenario, with_tile_list=True, max_attempts=2,
                        plan_first=False, self_check=False):
    feedback, best, first = "", None, None
    for attempt in range(1, max_attempts + 1):
        raw = chat_json(system_prompt(plan_first, self_check),
                        build_user_prompt(spatial_map, grid, scenario, with_tile_list, feedback))
        steps = raw.get("trajectory") or []
        errors = audit_raw(steps, grid, spatial_map)
        n_err = sum(len(v) for k, v in errors.items())
        print(f"  LLM attempt {attempt}: {len(steps)} steps, {n_err} rule violations")
        first = first or errors
        if best is None or n_err < best[2]:
            best = (raw.get("narrative", ""), steps, n_err, errors, raw.get("plan"))
        if n_err == 0 or not steps:
            break
        listed = [m for v in errors.values() for m in v][:12]
        feedback = ("\n\nYour previous answer broke these rules:\n- " + "\n- ".join(listed) +
                    "\nWrite the whole trajectory again without these errors.")
    return best[0], best[1], best[3], first, attempt, best[4]


def _pos(step):
    try:
        return int(round(float(step["position"][0]))), int(round(float(step["position"][1])))
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _facing(grid, spatial_map, pos, target):
    char = FURNITURE[target]["char"]
    return {name for name, (dx, dy) in ORIENTATIONS.items()
            if 0 <= pos[1] + dy < len(grid) and 0 <= pos[0] + dx < len(grid[0])
            and grid[pos[1] + dy][pos[0] + dx] == char}


def _sound_label(step):
    s = str(step.get("sound", "")).strip().lower().replace(" ", "_")
    return None if s in ("none", "", "silence", "null") else s


def audit_raw(steps, grid, spatial_map):
    err = {"not_walkable": [], "jump": [], "moving_non_walk": [], "bad_interaction": [],
           "bad_start": [], "bad_orientation": [], "bad_sound": []}
    furn, prev = spatial_map["furniture_metadata"], None
    for i, s in enumerate(steps):
        p, n = _pos(s), s.get("step", i + 1)
        if p is None:
            err["not_walkable"].append(f"step {n}: position missing or malformed")
            continue
        if i == 0 and list(p) != list(spatial_map["agent_start"]):
            err["bad_start"].append(f"step {n}: starts at {list(p)}, not at the actor start "
                                    f"{spatial_map['agent_start']}")
        if not is_walkable(grid, *p):
            err["not_walkable"].append(f"step {n}: {list(p)} is not a walkable tile")
        act = normalise_action(s.get("action", ""), s.get("description", ""))
        orient = str(s.get("orientation", "")).strip().lower()
        if prev is not None:
            d = abs(p[0] - prev[0]) + abs(p[1] - prev[1])
            if d > 1:
                err["jump"].append(f"step {n}: jumps {d} tiles from {list(prev)} to {list(p)}")
            if d >= 1 and act != "walking":
                err["moving_non_walk"].append(f"step {n}: position changes during '{act}'")
            if d == 1 and orient != orientation_between(prev, p):
                err["bad_orientation"].append(f"step {n}: moves {orientation_between(prev, p)} "
                                              f"but orientation is '{orient}'")
        tgt = s.get("target")
        if act != "walking" and isinstance(tgt, str) and tgt in furn:
            if p not in access_tiles(grid, spatial_map, tgt):
                err["bad_interaction"].append(f"step {n}: '{act}' at {list(p)} is not next to {tgt}")
            elif orient not in _facing(grid, spatial_map, p, tgt):
                err["bad_orientation"].append(f"step {n}: uses {tgt} but faces '{orient}', away from it")
        elif act != "walking" and ACTIONS[act]["needs"]:
            err["bad_interaction"].append(f"step {n}: '{act}' has no valid furniture target")
        snd = _sound_label(s)
        if snd is not None and snd not in SOUND_CLASSES:
            err["bad_sound"].append(f"step {n}: unknown sound '{snd}'")
        elif snd != ACTIONS[act]["sound"]:
            err["bad_sound"].append(f"step {n}: sound '{snd}' does not fit action '{act}' "
                                    f"(expected '{ACTIONS[act]['sound']}')")
        prev = p
    return err


def _walk_step(pos):
    return {"position": list(pos), "action": "walking", "target": None, "sound": "footsteps",
            "description": "Walking", "inserted": True}


def _infer_target(action, pos, tgt, spatial_map):
    furn = spatial_map["furniture_metadata"]
    needs = [k for k in ACTIONS[action]["needs"] if k in furn]
    if not needs:
        return None
    if isinstance(tgt, str) and tgt in furn:
        return tgt
    return min(needs, key=lambda k: abs(furn[k]["center_pos"][0] - pos[0] - 0.5)
               + abs(furn[k]["center_pos"][1] - pos[1] - 0.5))


def repair_trajectory(raw_steps, grid, spatial_map):
    rep = {"snapped": 0, "start_fixed": 0, "interaction_moved": 0, "bridged_gaps": 0,
           "bridge_steps_inserted": 0, "split_moving_actions": 0, "loop_tiles_removed": 0,
           "dropped": 0, "action_renamed": 0, "orientation_fixed": 0, "sound_replaced": 0}
    start = tuple(spatial_map["agent_start"])

    # snap positions to the floor and normalise actions and sounds
    steps = []
    for s in raw_steps:
        p = _pos(s)
        q = nearest_walkable(grid, *p) if p is not None else None
        if q is None:
            rep["dropped"] += 1
            continue
        rep["snapped"] += q != p
        act = normalise_action(s.get("action", ""), s.get("description", ""))
        rep["action_renamed"] += act != str(s.get("action", "")).strip().lower()
        snd = _sound_label(s)
        if snd is not None and snd not in SOUND_CLASSES:
            snd = ACTIONS[act]["sound"]
            rep["sound_replaced"] += 1
        steps.append({"position": list(q), "action": act, "target": s.get("target"),
                      "orientation": str(s.get("orientation", "")).strip().lower(),
                      "sound": snd or "none", "description": s.get("description", ""),
                      "duration_s": s.get("duration_s"), "inserted": False})
    if not steps or tuple(steps[0]["position"]) != start or steps[0]["action"] != "standing":
        rep["start_fixed"] = int(not steps or tuple(steps[0]["position"]) != start)
        steps.insert(0, {"position": list(start), "action": "standing", "target": None,
                         "orientation": "", "sound": "none",
                         "description": "Actor at the start position", "duration_s": 1.0, "inserted": True})

    # move interactions next to their furniture
    cur = start
    for s in steps:
        if s["action"] != "walking":
            s["target"] = _infer_target(s["action"], s["position"], s["target"], spatial_map)
            if s["target"]:
                acc = access_tiles(grid, spatial_map, s["target"])
                if acc and tuple(s["position"]) not in acc:
                    best = min(acc, key=lambda t: (len(astar(grid, cur, t)) or 10 ** 6, t))
                    s["position"] = list(best)
                    rep["interaction_moved"] += 1
        else:
            s["target"] = None
        cur = tuple(s["position"])

    # fill gaps with walking steps along the shortest path
    out = [steps[0]]
    for s in steps[1:]:
        a, b = tuple(out[-1]["position"]), tuple(s["position"])
        if a != b:
            path = astar(grid, a, b)
            if not path:
                rep["dropped"] += 1
                continue
            inter = path[1:] if s["action"] != "walking" else path[1:-1]
            if len(path) > 2:
                rep["bridged_gaps"] += 1
                rep["bridge_steps_inserted"] += len(path) - 2
            if s["action"] != "walking":
                rep["split_moving_actions"] += 1
            out.extend(_walk_step(p) for p in inter)
        elif s["action"] == "walking":
            continue
        out.append(s)

    # remove back-and-forth loops inside a walk
    final, run = [out[0]], []

    def flush():
        anchor, kept = tuple(final[-1]["position"]), []
        for w in run:
            p = tuple(w["position"])
            trail = [anchor] + [tuple(k["position"]) for k in kept]
            if p in trail:
                cut = trail.index(p)
                rep["loop_tiles_removed"] += len(kept) - cut + 1
                kept = kept[:cut]
            else:
                kept.append(w)
        final.extend(kept)
        run.clear()

    for s in out[1:]:
        if s["action"] == "walking":
            run.append(s)
        else:
            flush()
            a, b = tuple(final[-1]["position"]), tuple(s["position"])
            if a != b:
                final.extend(_walk_step(p) for p in astar(grid, a, b)[1:])
            final.append(s)
    flush()

    # fix orientations and renumber
    for i, s in enumerate(final):
        prev = final[i - 1]["position"] if i else s["position"]
        given = s.get("orientation", "")
        if s["action"] == "walking":
            good = {orientation_between(prev, s["position"])}
        elif s.get("target"):
            good = _facing(grid, spatial_map, tuple(s["position"]), s["target"])
        else:
            good = set(ORIENTATIONS)
        if given not in good:
            rep["orientation_fixed"] += not s.get("inserted", False)
            fallback = final[i - 1]["orientation"] if i else "south"
            s["orientation"] = sorted(good)[0] if len(good) < 4 else fallback
        s["step"] = i + 1
    assert_valid(final, grid, spatial_map)
    return final, rep


def assert_valid(steps, grid, spatial_map):
    assert list(steps[0]["position"]) == list(spatial_map["agent_start"]), "wrong start tile"
    for i, s in enumerate(steps):
        p = tuple(s["position"])
        assert is_walkable(grid, *p), f"step {i + 1}: {p} not walkable"
        if i:
            q = tuple(steps[i - 1]["position"])
            d = abs(p[0] - q[0]) + abs(p[1] - q[1])
            assert d <= 1, f"step {i + 1}: jump of {d} tiles"
            assert (d == 1) == (s["action"] == "walking"), f"step {i + 1}: movement/action mismatch"
        if s["action"] != "walking" and s.get("target"):
            assert p in access_tiles(grid, spatial_map, s["target"]), f"step {i + 1}: not at {s['target']}"


def add_timing(steps, scale_m, walk_speed=WALK_SPEED):
    t = 0.0
    for s in steps:
        if s["action"] == "walking":
            dur = scale_m / walk_speed
        else:
            try:
                dur = float(s.get("duration_s"))
            except (TypeError, ValueError):
                dur = ACTIONS[s["action"]]["default_s"]
            dur = min(max(dur, 1.0), 10.0)
        s["t_start"], s["duration_s"] = round(t, 4), round(dur, 4)
        t += dur
    return t


def build_timeline(steps):
    events, i = [], 0
    while i < len(steps):
        snd, j = steps[i]["sound"], i
        while j + 1 < len(steps) and steps[j + 1]["sound"] == snd and (
                snd == "footsteps" or steps[j + 1]["action"] == steps[i]["action"]):
            j += 1
        if snd != "none":
            events.append({"id": len(events), "sound_class": snd, "class_index": SOUND_CLASSES[snd]["index"],
                           "action": steps[i]["action"], "target": steps[i].get("target"),
                           "onset_s": steps[i]["t_start"],
                           "offset_s": round(steps[j]["t_start"] + steps[j]["duration_s"], 4),
                           "steps": [i + 1, j + 1]})
        i = j + 1
    return events


def trajectory_metrics(raw_steps, audit, steps, grid):
    n = max(len(raw_steps), 1)
    moves = max(len(raw_steps) - 1, 1)
    stationary = sum(1 for s in raw_steps if normalise_action(s.get("action", ""), s.get("description", ""))
                     != "walking")

    order = []
    for s in raw_steps:
        t = s.get("target")
        if isinstance(t, str) and t in FURNITURE and (not order or order[-1] != t):
            order.append(t)

    anchors = [tuple(steps[0]["position"])] + [tuple(s["position"]) for s in steps if s["action"] != "walking"]
    anchors.append(tuple(steps[-1]["position"]))
    shortest = sum(max(len(astar(grid, a, b)) - 1, 0) for a, b in zip(anchors[:-1], anchors[1:]))
    walked = sum(1 for s in steps if s["action"] == "walking")
    return {
        "raw_steps": len(raw_steps),
        "violations_total": sum(len(v) for v in audit.values()),
        "fully_valid": int(sum(len(v) for v in audit.values()) == 0),
        "pct_positions_walkable": round(100 * (1 - len(audit["not_walkable"]) / n), 1),
        "pct_moves_one_tile": round(100 * (1 - len(audit["jump"]) / moves), 1),
        "pct_orientation_ok": round(100 * (1 - len(audit["bad_orientation"]) / n), 1),
        "pct_sound_ok": round(100 * (1 - len(audit["bad_sound"]) / n), 1),
        "pct_interactions_ok": round(100 * (1 - len(audit["bad_interaction"]) / max(stationary, 1)), 1),
        "moving_non_walk": len(audit["moving_non_walk"]),
        "starts_at_start": int(not audit["bad_start"]),
        "targets_in_order": order,
        "final_steps": len(steps),
        "path_efficiency": round(shortest / walked, 3) if walked else 1.0,
    }


def run(spatial_map, scenario, with_tile_list=True, max_attempts=2, plan_first=False, self_check=False):
    grid = build_grid(spatial_map)
    narrative, raw, audit, first_audit, attempts, plan = generate_trajectory(
        spatial_map, grid, scenario, with_tile_list, max_attempts, plan_first, self_check)
    if not raw:
        raise RuntimeError("the LLM returned no trajectory")
    steps, report = repair_trajectory(raw, grid, spatial_map)
    total = add_timing(steps, spatial_map["grid"]["scale_m"])
    return {
        "scenario": scenario, "narrative": narrative, "source": "llm",
        "prompt": {"walkable_tile_list": with_tile_list, "plan_first": plan_first,
                   "self_check": self_check, "attempts_used": attempts},
        "llm_plan": plan,
        "duration_s": round(total, 4), "walk_speed_mps": WALK_SPEED,
        "raw_trajectory": raw,
        "raw_violations": {k: len(v) for k, v in audit.items()},
        "raw_violation_details": audit,
        "first_attempt_violations": {k: len(v) for k, v in first_audit.items()},
        "repair_report": report,
        "metrics": trajectory_metrics(raw, audit, steps, grid),
        "trajectory": steps,
        "sound_timeline": build_timeline(steps),
    }


def render_path_ascii(steps, grid):
    g = [row[:] for row in grid]
    for s in steps:
        x, y = s["position"]
        if g[y][x] == "_":
            g[y][x] = "."
    marks = []
    for s in steps:
        if s["action"] != "walking" and (not marks or marks[-1][1] != tuple(s["position"])):
            marks.append((s["action"], tuple(s["position"])))
    for n, (_, (x, y)) in enumerate(marks):
        g[y][x] = str(n % 10)
    return ascii_with_axes(g) + "\n    " + ", ".join(f"{n % 10}={a}" for n, (a, _) in enumerate(marks))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 2: Action Scripter")
    ap.add_argument("--map", default="spatial_map.json")
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO)
    ap.add_argument("--no-tile-list", action="store_true", help="prompt without the walkable tile list")
    ap.add_argument("--plan-first", action="store_true", help="ask the LLM to list its waypoints before the steps")
    ap.add_argument("--self-check", action="store_true", help="ask the LLM to verify its steps before answering")
    ap.add_argument("--attempts", type=int, default=2, help="LLM attempts (1 = no feedback round)")
    ap.add_argument("--out", default="trajectory.json")
    args = ap.parse_args()

    sm = load_json(args.map)
    result = run(sm, args.scenario, with_tile_list=not args.no_tile_list, max_attempts=args.attempts,
                 plan_first=args.plan_first, self_check=args.self_check)

    print(f"\nScenario  : {result['scenario']}\nNarrative : {result['narrative']}")
    print(f"LLM answer: {len(result['raw_trajectory'])} steps, violations {result['raw_violations']}")
    for msgs in result["raw_violation_details"].values():
        for m in msgs[:4]:
            print("    -", m)
    print(f"Repairs   : { {k: v for k, v in result['repair_report'].items() if v} or 'none needed'}")
    print(f"Final     : {len(result['trajectory'])} steps, {result['duration_s']:.1f} s, "
          f"path efficiency {result['metrics']['path_efficiency']}\n")
    print(render_path_ascii(result["trajectory"], build_grid(sm)))
    print("\nSOUND EVENT TIMELINE")
    for e in result["sound_timeline"]:
        print(f"  {e['onset_s']:6.2f} to {e['offset_s']:6.2f} s  {e['sound_class']:10s} ({e['action']}"
              f"{', at ' + e['target'] if e['target'] else ''})")
    save_json(result, args.out)
    print(f"\nSaved {args.out}")

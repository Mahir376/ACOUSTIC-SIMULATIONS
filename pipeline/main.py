"""Stage 1: room generation. Reads a scenario, asks the LLM for a room layout and writes spatial_map.json."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import deque

from scene_common import (
    CEILING_M, CEILING_MATERIALS, DEFAULT_SURFACES, FLOOR_MATERIALS, FURNITURE,
    GRID_H, GRID_W, PLACEMENT_ORDER, SCALE_M, WALL_MATERIALS,
    ascii_with_axes, build_base_grid, build_grid, chat_json, normalise_polygon,
    grid_to_ascii, polygon_area, save_json, spatial_map_from_ascii,
)

SHAPES = ["rectangular", "L-shaped", "T-shaped"]

EXAMPLE_ROOMS = {
    "rectangular": [[1, 1], [23, 1], [23, 13], [1, 13]],
    "L-shaped": [[1, 1], [13, 1], [13, 7], [23, 7], [23, 13], [1, 13]],
    "T-shaped": [[1, 5], [8, 5], [8, 1], [15, 1], [15, 5], [23, 5], [23, 13], [1, 13]],
}


def load_references(folder: str):
    refs = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            sm = spatial_map_from_ascii(f.read())
        refs.append((os.path.splitext(os.path.basename(path))[0], sm))
    if not refs:
        raise FileNotFoundError(f"no .txt tile maps found in '{folder}'")
    return refs


def references_text(references) -> str:
    out = ["REFERENCE LAYOUTS",
           "These rooms were traced from real floor plans (one tile = 0.5 m, W = wall, _ = floor,",
           "K kitchen counter, S sofa, V television, B bed, T table). Study how real flats arrange",
           "furniture: what stands against a wall, what faces what, how far apart things are and how",
           "much open floor is left. Design a NEW room in the same spirit. Do not copy a reference,",
           "and keep the room size, shape and furniture sizes given above."]
    for i, (name, sm) in enumerate(references, 1):
        furn = ", ".join(f"{k} at {m['top_left']} size {m['size'][0]}x{m['size'][1]}"
                         for k, m in sm["furniture_metadata"].items())
        out += ["", f"Reference {i} ({sm['floor_area_m2']:.0f} m2):",
                grid_to_ascii(build_grid({**sm, "mic_pos": None}, with_mic=False)),
                f"Furniture (top-left tile, width x height): {furn}"]
    return "\n".join(out)


PLACEMENT_RULES = """PLACEMENT RULES (how real flats are furnished)
- The kitchen counter, the bed, the sofa and the television each stand against a wall.
- The television faces the sofa from 2 to 4 m away (4 to 8 tiles), with free floor between them.
- The bed is in the part of the room farthest from the kitchen counter.
- Leave walking space: at least one free tile in front of every item."""

REASONING_STEP = """THINK FIRST
Before giving any coordinates, fill the "reasoning" field: name the zones of the flat (cooking,
dining, living, sleeping), say which part of the room each zone occupies and why, and where each
furniture item goes inside its zone. Then give coordinates that match this reasoning."""


def system_prompt(with_examples: bool = True, references=None, rules=False, reasoning=False) -> str:
    extra = (PLACEMENT_RULES + "\n\n" if rules else "") + (REASONING_STEP + "\n\n" if reasoning else "")
    reasoning_field = '\n  "reasoning": "zones and placement, in a few sentences",' if reasoning else ""
    p = f"""You are an architectural designer who draws small flats on a tile grid.

GRID
- {GRID_W} columns (x = 0..{GRID_W - 1}, left to right) and {GRID_H} rows (y = 0..{GRID_H - 1}, top to bottom).
- One tile is {SCALE_M} m x {SCALE_M} m.
- Room corners lie on tile boundaries and must satisfy 1 <= x <= {GRID_W - 1} and 1 <= y <= {GRID_H - 1}.

ROOM RULES
1. "room_corners" is ONE closed polygon, listed in order, with 4 to 10 vertices.
2. Every edge is horizontal or vertical.
3. The polygon must not cross itself and must cover at least 60 tiles.

FURNITURE (give the [x, y] tile of the TOP-LEFT corner of each item)
- KK   kitchen counter, 2x2 tiles
- SSSS sofa, 4x2 tiles
- TV   television, 2x1 tiles
- BB   bed, 2x4 tiles
- TTTT table, 4x2 tiles
All furniture tiles must be inside the room, must not overlap, and the layout
should be one that a real flat could have.

OTHER FIELDS
- "mic_pos": a free floor tile for a fixed microphone, in open space.
- "agent_start": a free floor tile at least 3 tiles from the microphone.
- "surfaces": choose one material each from
    walls   {WALL_MATERIALS}
    floor   {FLOOR_MATERIALS}
    ceiling {CEILING_MATERIALS}

{extra}Return ONLY this JSON object:
{{{reasoning_field}
  "room_corners": [[x, y], ...],
  "furniture": {{"KK": [x, y], "SSSS": [x, y], "TV": [x, y], "BB": [x, y], "TTTT": [x, y]}},
  "mic_pos": [x, y],
  "agent_start": [x, y],
  "surfaces": {{"walls": "...", "floor": "...", "ceiling": "..."}}
}}"""
    if with_examples:
        p += "\n\nEXAMPLE POLYGONS\n" + "\n".join(f"- {k}: {v}" for k, v in EXAMPLE_ROOMS.items())
    if references:
        p += "\n\n" + references_text(references)
    return p


def propose_room(scenario: str, with_examples=True, max_attempts=3, references=None,
                 rules=False, reasoning=False):
    feedback, log = "", []
    for attempt in range(1, max_attempts + 1):
        raw = chat_json(system_prompt(with_examples, references, rules, reasoning),
                        f"SCENARIO: {scenario}{feedback}")
        corners, problems = normalise_polygon(raw.get("room_corners"))
        log.append({"attempt": attempt, "room_corners": raw.get("room_corners"), "problems": problems})
        if not problems:
            raw["room_corners_raw"] = raw.get("room_corners")
            raw["room_corners"] = corners
            return raw, log
        print(f"  attempt {attempt}: polygon rejected: {problems}")
        feedback = ("\n\nYour previous answer was rejected for these reasons: "
                    + "; ".join(problems) + ". Fix them and answer again.")
    return None, log


def _floor_connected(grid) -> bool:
    tiles = [(x, y) for y, r in enumerate(grid) for x, c in enumerate(r) if c == "_"]
    if not tiles:
        return False
    seen, q = {tiles[0]}, deque([tiles[0]])
    while q:
        cx, cy = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (cx + dx, cy + dy)
            if p not in seen and grid[p[1]][p[0]] == "_":
                seen.add(p)
                q.append(p)
    return len(seen) == len(tiles)


def _block(x, y, w, h):
    return [(x + dx, y + dy) for dy in range(h) for dx in range(w)]


def _wall_contact(grid, cells):
    return sum(any(grid[cy + dy][cx + dx] == "W" for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
               for cx, cy in cells)


def _fits(grid, cells):
    h, w = len(grid), len(grid[0])
    if any(not (0 <= cy < h and 0 <= cx < w) or grid[cy][cx] != "_" for cx, cy in cells):
        return False
    s = set(cells)
    if not any((cx + dx, cy + dy) not in s and grid[cy + dy][cx + dx] == "_"
               for cx, cy in cells for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
        return False
    for cx, cy in cells:
        grid[cy][cx] = "#"
    ok = _floor_connected(grid)
    for cx, cy in cells:
        grid[cy][cx] = "_"
    return ok


def place_furniture(grid, llm_furniture, report):
    meta, n_valid = {}, 0
    for key in PLACEMENT_ORDER:
        w0, h0 = FURNITURE[key]["size"]
        try:
            llm = (int(llm_furniture[key][0]), int(llm_furniture[key][1]))
        except (TypeError, ValueError, IndexError, KeyError):
            llm = None
        chosen = None
        if llm is not None:
            for w, h in ((w0, h0), (h0, w0)):
                if _fits(grid, _block(*llm, w, h)):
                    chosen = (*llm, w, h)
                    n_valid += 1
                    break
        if chosen is None:
            ref = llm or (len(grid[0]) // 2, len(grid) // 2)
            best = None
            for w, h in {(w0, h0), (h0, w0)}:
                for y in range(len(grid) - h + 1):
                    for x in range(len(grid[0]) - w + 1):
                        cells = _block(x, y, w, h)
                        if not _fits(grid, cells):
                            continue
                        cost = abs(x - ref[0]) + abs(y - ref[1])
                        if best is None or cost < best[0]:
                            best = (cost, x, y, w, h)
            if best is None:
                report.append(f"{key}: no valid position exists, item left out")
                continue
            chosen = best[1:]
            report.append(f"{key}: LLM position {list(llm) if llm else 'missing'} invalid, "
                          f"moved to {[chosen[0], chosen[1]]}")
        x, y, w, h = chosen
        for cx, cy in _block(x, y, w, h):
            grid[cy][cx] = FURNITURE[key]["char"]
        meta[key] = {"top_left": [x, y], "size": [w, h], "center_pos": [x + w / 2, y + h / 2],
                     "label": FURNITURE[key]["label"], "material": FURNITURE[key]["material"],
                     "height_m": FURNITURE[key]["height"]}
    return meta, n_valid


def _clearance(grid, x, y, limit=4):
    for r in range(1, limit + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                inside = 0 <= y + dy < len(grid) and 0 <= x + dx < len(grid[0])
                if not inside or grid[y + dy][x + dx] != "_":
                    return r - 1
    return limit


def _as_tile(p):
    try:
        return int(p[0]), int(p[1])
    except (TypeError, ValueError, IndexError):
        return None


def choose_mic(grid, llm_pos, report):
    free = [(x, y) for y, r in enumerate(grid) for x, c in enumerate(r) if c == "_"]
    ref = _as_tile(llm_pos)
    ok = ref in free
    if ok:
        grid[ref[1]][ref[0]] = "M"
        ok = _floor_connected(grid)
        if ok:
            return list(ref), True
        grid[ref[1]][ref[0]] = "_"
    best_clear = max(min(_clearance(grid, x, y), 2) for x, y in free)
    cands = [p for p in free if _clearance(grid, *p) >= best_clear]
    if ref is None:
        ref = (sum(x for x, _ in free) / len(free), sum(y for _, y in free) / len(free))
    for p in sorted(cands, key=lambda p: (abs(p[0] - ref[0]) + abs(p[1] - ref[1]), p)):
        grid[p[1]][p[0]] = "M"
        if _floor_connected(grid):
            report.append(f"mic_pos: LLM position {llm_pos} invalid, moved to {list(p)}")
            return list(p), False
        grid[p[1]][p[0]] = "_"
    raise RuntimeError("no valid microphone tile")


def choose_agent_start(grid, llm_pos, mic, report, min_dist=3):
    free = [(x, y) for y, r in enumerate(grid) for x, c in enumerate(r) if c == "_"]
    far = [p for p in free if abs(p[0] - mic[0]) + abs(p[1] - mic[1]) >= min_dist] or free
    ref = _as_tile(llm_pos)
    if ref in far:
        return list(ref), True
    if ref is None:
        ref = max(far, key=lambda p: abs(p[0] - mic[0]) + abs(p[1] - mic[1]))
    start = min(far, key=lambda p: (abs(p[0] - ref[0]) + abs(p[1] - ref[1]), p))
    report.append(f"agent_start: LLM position {llm_pos} invalid, moved to {list(start)}")
    return list(start), False


def layout_metrics(spatial_map):
    grid = build_grid(spatial_map, with_mic=False)
    scale = spatial_map["grid"]["scale_m"]
    furn = spatial_map["furniture_metadata"]
    dist = lambda a, b: scale * ((furn[a]["center_pos"][0] - furn[b]["center_pos"][0]) ** 2 +
                                 (furn[a]["center_pos"][1] - furn[b]["center_pos"][1]) ** 2) ** 0.5
    wall_items = [k for k in ("KK", "BB", "SSSS", "TV") if k in furn]
    at_wall = sum(_wall_contact(grid, _block(*furn[k]["top_left"], *furn[k]["size"])) > 0 for k in wall_items)
    return {
        "floor_area_m2": round(spatial_map["floor_area_m2"], 2),
        "n_corners": len(spatial_map["room_corners"]),
        "n_furniture": len(furn),
        "wall_items_at_wall": f"{at_wall}/{len(wall_items)}",
        "tv_sofa_distance_m": round(dist("TV", "SSSS"), 2) if "TV" in furn and "SSSS" in furn else None,
        "tv_sofa_2_to_4_m": int(2.0 <= dist("TV", "SSSS") <= 4.0) if "TV" in furn and "SSSS" in furn else None,
        "bed_kitchen_distance_m": round(dist("BB", "KK"), 2) if "BB" in furn and "KK" in furn else None,
        "table_kitchen_distance_m": round(dist("TTTT", "KK"), 2) if "TTTT" in furn and "KK" in furn else None,
    }


def generate_spatial_map(scenario: str, with_examples=True, max_attempts=3, references=None,
                         rules=False, reasoning=False):
    raw, llm_log = propose_room(scenario, with_examples, max_attempts, references, rules, reasoning)
    if raw is None:
        return None, llm_log
    report = []
    corners = raw["room_corners"]
    grid = build_base_grid(corners, GRID_W, GRID_H)
    furniture, n_furn_valid = place_furniture(grid, raw.get("furniture") or {}, report)
    mic, mic_ok = choose_mic(grid, raw.get("mic_pos"), report)
    start, start_ok = choose_agent_start(grid, raw.get("agent_start"), mic, report)

    surfaces, surf_ok = dict(DEFAULT_SURFACES), 0
    for name, allowed in (("walls", WALL_MATERIALS), ("floor", FLOOR_MATERIALS), ("ceiling", CEILING_MATERIALS)):
        choice = (raw.get("surfaces") or {}).get(name)
        if choice in allowed:
            surfaces[name] = choice
            surf_ok += 1
        else:
            report.append(f"surfaces.{name}: '{choice}' is not allowed, using '{surfaces[name]}'")

    sm = {
        "scenario": scenario, "source": "llm",
        "prompt": {"example_polygons": with_examples, "placement_rules": rules, "reasoning_first": reasoning,
                   "reference_layouts": [name for name, _ in references] if references else []},
        "grid": {"width": GRID_W, "height": GRID_H, "scale_m": SCALE_M, "ceiling_m": CEILING_M},
        "room_corners": corners,
        "floor_area_m2": abs(polygon_area(corners)) * SCALE_M ** 2,
        "surfaces": surfaces, "furniture_metadata": furniture,
        "mic_pos": mic, "agent_start": start,
        "raw_layout": {
            "polygon_valid_first_try": not llm_log[0]["problems"],
            "polygon_attempts": len(llm_log),
            "furniture_valid": n_furn_valid, "furniture_total": len(FURNITURE),
            "mic_valid": mic_ok, "agent_start_valid": start_ok, "surfaces_valid": surf_ok,
            "llm_answer": {k: raw.get(k) for k in ("reasoning", "room_corners_raw", "furniture", "mic_pos",
                                                   "agent_start", "surfaces")},
        },
        "validation_report": report, "llm_attempts": llm_log,
    }
    sm["layout_metrics"] = layout_metrics(sm)
    sm["ascii_map"] = ascii_with_axes(build_grid(sm))
    return sm, llm_log


def load_given_map(path: str):
    with open(path, "r", encoding="utf-8") as f:
        sm = spatial_map_from_ascii(f.read())
    report = []
    grid = build_grid({**sm, "mic_pos": None}, with_mic=False)
    sm["mic_pos"], _ = choose_mic(grid, sm["mic_pos"], report)
    sm["agent_start"], _ = choose_agent_start(grid, sm["agent_start"], sm["mic_pos"], report)
    sm.update({"scenario": f"given map {path}", "source": "given", "surfaces": dict(DEFAULT_SURFACES),
               "validation_report": report, "llm_attempts": []})
    sm["layout_metrics"] = layout_metrics(sm)
    sm["ascii_map"] = ascii_with_axes(build_grid(sm))
    return sm


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 1: Spatial Architect")
    ap.add_argument("--shape", choices=SHAPES, default="T-shaped")
    ap.add_argument("--scenario", default=None, help="free text; overrides --shape")
    ap.add_argument("--no-examples", action="store_true", help="prompt without the example polygons")
    ap.add_argument("--references", default=None, metavar="FOLDER",
                    help="folder of tile maps traced from real floor plans, shown to the LLM as references")
    ap.add_argument("--rules", action="store_true", help="add explicit furniture placement rules to the prompt")
    ap.add_argument("--reasoning", action="store_true", help="ask the LLM to describe zones before coordinates")
    ap.add_argument("--given-map", default=None, help="hand-drawn tile map file; skips the LLM")
    ap.add_argument("--out", default="spatial_map.json")
    args = ap.parse_args()

    if args.given_map:
        sm = load_given_map(args.given_map)
    else:
        scenario = args.scenario or f"A {args.shape} studio flat with a kitchen corner, a living area and a bed."
        refs = load_references(args.references) if args.references else None
        if refs:
            print(f"Prompt includes {len(refs)} reference layouts from {args.references}")
        sm, log = generate_spatial_map(scenario, with_examples=not args.no_examples, references=refs,
                                       rules=args.rules, reasoning=args.reasoning)
        if sm is None:
            print("\nThe LLM did not produce a valid room polygon in 3 attempts:")
            for a in log:
                print(f"  attempt {a['attempt']}: {a['room_corners']}  ->  {a['problems']}")
            sys.exit(1)

    print(f"\nScenario : {sm['scenario']}   (source: {sm['source']})")
    print(f"Layout   : {sm['layout_metrics']}")
    print(f"Surfaces : {sm['surfaces']}")
    print("\n" + sm["ascii_map"])
    print(f"\nMicrophone M at tile {sm['mic_pos']}, actor starts at tile {sm['agent_start']}")
    if sm["source"] == "llm":
        r = sm["raw_layout"]
        if r["llm_answer"].get("reasoning"):
            print(f"LLM reasoning : {r['llm_answer']['reasoning']}")
        print(f"LLM on its own: polygon valid at first try: {r['polygon_valid_first_try']}, "
              f"furniture valid {r['furniture_valid']}/{r['furniture_total']}, "
              f"mic valid: {r['mic_valid']}, start valid: {r['agent_start_valid']}")
    for line in sm["validation_report"]:
        print("  corrected:", line)
    save_json(sm, args.out)
    print(f"\nSaved {args.out}")

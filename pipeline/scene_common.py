"""Shared grid, coordinate, map and LLM helpers used by all stages."""

from __future__ import annotations

import heapq
import json
import os
from collections import deque

import numpy as np
from matplotlib.path import Path


GRID_W = 24
GRID_H = 14
SCALE_M = 0.5
CEILING_M = 2.6
FS = 24000
WALK_SPEED = 1.25
MIC_HEIGHT = 1.2
SOURCE_HEIGHT = 1.5

OCTAVE_BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]


FURNITURE = {
    "KK":   {"size": (2, 2), "char": "K", "label": "kitchen counter",
             "material": "marble_floor",              "height": 0.90},
    "BB":   {"size": (2, 4), "char": "B", "label": "bed",
             "material": "chairs_heavy_upholstered",  "height": 0.50},
    "SSSS": {"size": (4, 2), "char": "S", "label": "sofa",
             "material": "chairs_medium_upholstered", "height": 0.80},
    "TTTT": {"size": (4, 2), "char": "T", "label": "table",
             "material": "wood_16mm",                 "height": 0.75},
    "TV":   {"size": (2, 1), "char": "V", "label": "television",
             "material": "glass_3mm",                 "height": 1.00},
}
FURN_CHARS = {v["char"]: k for k, v in FURNITURE.items()}
PLACEMENT_ORDER = ["KK", "SSSS", "TV", "BB", "TTTT"]


WALL_MATERIALS = ["plasterboard", "brickwork", "rough_concrete", "wooden_lining"]
FLOOR_MATERIALS = ["stage_floor", "linoleum_on_concrete", "carpet_thin",
                   "carpet_soft_10mm", "ceramic_tiles"]
CEILING_MATERIALS = ["ceiling_plasterboard", "plasterboard", "ceiling_fissured_tile"]
DEFAULT_SURFACES = {"walls": "plasterboard", "floor": "stage_floor",
                    "ceiling": "ceiling_plasterboard"}


ACTIONS = {
    "walking":       {"sound": "footsteps", "default_s": None, "needs": []},
    "standing":      {"sound": None,        "default_s": 1.5,  "needs": []},
    "running_water": {"sound": "water_tap", "default_s": 4.0,  "needs": ["KK"]},
    "cooking":       {"sound": "cooking",   "default_s": 6.0,  "needs": ["KK"]},
    "eating":        {"sound": "cutlery",   "default_s": 6.0,  "needs": ["TTTT", "SSSS"]},
    "typing":        {"sound": "keyboard",  "default_s": 5.0,  "needs": ["TTTT"]},
    "sitting_down":  {"sound": "chair",     "default_s": 1.5,  "needs": ["SSSS", "TTTT", "BB"]},
    "watching_tv":   {"sound": "tv",        "default_s": 6.0,  "needs": ["SSSS", "TV"]},
    "speaking":      {"sound": "speech",    "default_s": 3.0,  "needs": []},
    "resting":       {"sound": None,        "default_s": 3.0,  "needs": ["SSSS", "BB"]},
}


ACTION_KEYWORDS = [
    ("walking",       ["walk", "move", "moving", "approach", "head", "go to", "goes", "return", "navigat", "step"]),
    ("running_water", ["water", "tap", "sink", "wash", "rinse", "faucet", "drink"]),
    ("cooking",       ["cook", "fry", "boil", "chop", "sizzl", "prepar", "stove"]),
    ("typing",        ["typ", "keyboard", "laptop", "computer", "work"]),
    ("eating",        ["eat", "cutlery", "fork", "knife", "meal", "dinner", "lunch", "breakfast"]),
    ("watching_tv",   ["tv", "television", "watch", "remote"]),
    ("sitting_down",  ["sit", "seat", "chair", "settle"]),
    ("speaking",      ["talk", "speak", "say", "call", "chat", "phone"]),
    ("resting",       ["rest", "relax", "sleep", "lie", "nap"]),
    ("standing",      ["stand", "wait", "pause", "look", "survey", "observe", "turn", "idle"]),
]


SOUND_CLASSES = {
    "footsteps": {"index": 0, "starss23": 6,    "mode": "steps",   "level_db": -30.0},
    "water_tap": {"index": 1, "starss23": 10,   "mode": "loop",    "level_db": -24.0},
    "cooking":   {"index": 2, "starss23": 5,    "mode": "loop",    "level_db": -26.0},
    "cutlery":   {"index": 3, "starss23": 5,    "mode": "loop",    "level_db": -28.0},
    "keyboard":  {"index": 4, "starss23": 5,    "mode": "loop",    "level_db": -32.0},
    "chair":     {"index": 5, "starss23": 5,    "mode": "oneshot", "level_db": -26.0},
    "tv":        {"index": 6, "starss23": 8,    "mode": "loop",    "level_db": -22.0},
    "speech":    {"index": 7, "starss23": 1,    "mode": "loop",    "level_db": -22.0},
}


SAMPLE_FILES = {
    "footsteps": ["footsteps.wav", "walking.wav"],
    "water_tap": ["water_tap.wav", "water.wav"],
    "cooking":   ["cooking.wav"],
    "cutlery":   ["cutlery.wav"],
    "keyboard":  ["keyboard.wav"],
    "chair":     ["chair.wav"],
    "tv":        ["tv.wav"],
    "speech":    ["speech.wav", "talking.wav"],
}


def chat_json(system_prompt: str, user_prompt: str) -> dict:
    from openai import AzureOpenAI

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not endpoint or not key:
        raise RuntimeError(
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in the environment "
            "before running main.py or action.py."
        )
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )
    response = client.chat.completions.create(
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat"),
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def tile_centre_m(x: float, y: float, scale=SCALE_M, grid_h=GRID_H):
    return ((x + 0.5) * scale, (grid_h - y - 0.5) * scale)


def grid_point_m(gx: float, gy: float, scale=SCALE_M, grid_h=GRID_H):
    return (gx * scale, (grid_h - gy) * scale)


def corners_m(spatial_map: dict) -> np.ndarray:
    g = spatial_map["grid"]
    pts = np.array([grid_point_m(x, y, g["scale_m"], g["height"])
                    for x, y in spatial_map["room_corners"]], dtype=float)
    if polygon_area(pts) < 0:
        pts = pts[::-1]
    return pts.T


def polygon_area(pts) -> float:
    pts = np.asarray(pts, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def furniture_centre_m(spatial_map: dict, key: str):
    g = spatial_map["grid"]
    meta = spatial_map["furniture_metadata"][key]
    tlx, tly = meta["top_left"]
    w, h = meta["size"]
    x, y = grid_point_m(tlx + w / 2, tly + h / 2, g["scale_m"], g["height"])
    return x, y, FURNITURE[key]["height"]


def az_el_dist(src_xyz, mic_xyz):
    d = np.asarray(src_xyz, float) - np.asarray(mic_xyz, float)
    dist = float(np.linalg.norm(d))
    az = float(np.degrees(np.arctan2(d[1], d[0])))
    el = float(np.degrees(np.arcsin(d[2] / dist))) if dist > 1e-9 else 0.0
    return az, el, dist


def _segments_cross(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return (v > 0) - (v < 0)

    def on_seg(a, b, c):
        return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
                and min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

    o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
    o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_seg(p1, p2, p3): return True
    if o2 == 0 and on_seg(p1, p2, p4): return True
    if o3 == 0 and on_seg(p3, p4, p1): return True
    if o4 == 0 and on_seg(p3, p4, p2): return True
    return False


def normalise_polygon(corners, width=GRID_W, height=GRID_H):
    problems = []
    pts = [(int(round(float(p[0]))), int(round(float(p[1])))) for p in corners or []]


    pts = [p for i, p in enumerate(pts) if p != pts[i - 1]] if len(pts) > 1 else pts
    if len(pts) < 4:
        return [], ["polygon needs at least 4 distinct vertices"]


    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    dx = max(0, 1 - min(xs)); dy = max(0, 1 - min(ys))
    dx -= max(0, max(xs) + dx - (width - 1)); dy -= max(0, max(ys) + dy - (height - 1))
    pts = [(min(max(x + dx, 1), width - 1), min(max(y + dy, 1), height - 1)) for x, y in pts]
    pts = [p for i, p in enumerate(pts) if p != pts[i - 1]]


    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) == 0:
                pts.pop(i); changed = True
                break
    if len(pts) < 4:
        return [], ["polygon is degenerate after cleaning"]

    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        if a[0] != b[0] and a[1] != b[1]:
            problems.append(f"edge {list(a)}->{list(b)} is not axis-aligned")
    for i in range(n):
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            if _segments_cross(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]):
                problems.append("polygon is self-intersecting")
                break
        else:
            continue
        break
    if abs(polygon_area(pts)) < 60:
        problems.append("room is smaller than 60 tiles (15 m^2)")
    return [list(p) for p in pts], problems


def build_base_grid(corners, width=GRID_W, height=GRID_H):
    poly = Path(corners)
    centres = [(x + 0.5, y + 0.5) for y in range(height) for x in range(width)]
    inside = poly.contains_points(centres).reshape(height, width)
    inside[0, :] = inside[-1, :] = False
    inside[:, 0] = inside[:, -1] = False


    seen = np.zeros_like(inside, dtype=bool)
    best = []
    for y in range(height):
        for x in range(width):
            if inside[y, x] and not seen[y, x]:
                comp, q = [], deque([(x, y)])
                seen[y, x] = True
                while q:
                    cx, cy = q.popleft()
                    comp.append((cx, cy))
                    for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = cx + ddx, cy + ddy
                        if 0 <= nx < width and 0 <= ny < height and inside[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((nx, ny))
                if len(comp) > len(best):
                    best = comp
    grid = [[" "] * width for _ in range(height)]
    for x, y in best:
        grid[y][x] = "_"
    for x, y in best:
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                nx, ny = x + ddx, y + ddy
                if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == " ":
                    grid[ny][nx] = "W"
    return grid


def build_grid(spatial_map: dict, with_mic=True):
    g = spatial_map["grid"]
    grid = build_base_grid(spatial_map["room_corners"], g["width"], g["height"])
    for key, meta in spatial_map.get("furniture_metadata", {}).items():
        tlx, tly = meta["top_left"]
        w, h = meta["size"]
        for dy in range(h):
            for dx in range(w):
                grid[tly + dy][tlx + dx] = FURNITURE[key]["char"]
    if with_mic and spatial_map.get("mic_pos"):
        mx, my = spatial_map["mic_pos"]
        grid[my][mx] = "M"
    return grid


def grid_to_ascii(grid) -> str:
    return "\n".join(" ".join(row) for row in grid)


def ascii_with_axes(grid) -> str:
    w = len(grid[0])
    head = "    " + " ".join(f"{x % 10}" for x in range(w))
    tens = "    " + " ".join(f"{x // 10}" if x >= 10 else " " for x in range(w))
    rows = [f"{y:2d}  " + " ".join(row) for y, row in enumerate(grid)]
    return "\n".join([tens, head] + rows)


WALKABLE = {"_"}


def is_walkable(grid, x, y) -> bool:
    return 0 <= y < len(grid) and 0 <= x < len(grid[0]) and grid[y][x] in WALKABLE


def walkable_tiles(grid):
    return [(x, y) for y, row in enumerate(grid) for x, c in enumerate(row) if c in WALKABLE]


def nearest_walkable(grid, x, y, forbid=()):
    if is_walkable(grid, x, y) and (x, y) not in forbid:
        return (x, y)
    seen, q = {(x, y)}, deque([(x, y)])
    limit = len(grid) * len(grid[0]) * 4
    while q and limit > 0:
        limit -= 1
        cx, cy = q.popleft()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in seen or not (-2 <= nx < len(grid[0]) + 2 and -2 <= ny < len(grid) + 2):
                continue
            seen.add((nx, ny))
            if is_walkable(grid, nx, ny) and (nx, ny) not in forbid:
                return (nx, ny)
            q.append((nx, ny))
    return None


def astar(grid, start, goal):
    start, goal = tuple(start), tuple(goal)
    if start == goal:
        return [start]
    open_set = [(0, 0, start)]
    came, g = {}, {start: 0}
    tie = 0
    while open_set:
        _, _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = (cur[0] + dx, cur[1] + dy)
            if not is_walkable(grid, *nb):
                continue
            t = g[cur] + 1
            if t < g.get(nb, 1e18):
                came[nb], g[nb] = cur, t
                tie += 1
                h = abs(nb[0] - goal[0]) + abs(nb[1] - goal[1])
                heapq.heappush(open_set, (t + h, tie, nb))
    return []


def access_tiles(grid, spatial_map, key):
    meta = spatial_map["furniture_metadata"][key]
    tlx, tly = meta["top_left"]
    w, h = meta["size"]
    out = set()
    for dy in range(h):
        for dx in range(w):
            for ax, ay in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (tlx + dx + ax, tly + dy + ay)
                if is_walkable(grid, *p):
                    out.add(p)
    return sorted(out)


def orientation_between(a, b, default="north"):
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return "east" if dx > 0 else "west"
    if dy != 0:
        return "south" if dy > 0 else "north"
    return default


def material_coeffs(name: str) -> np.ndarray:
    import pyroomacoustics as pra
    ea = pra.Material(name).energy_absorption
    c, f = np.atleast_1d(ea["coeffs"]).astype(float), ea.get("center_freqs")
    if f is None or len(c) == 1:
        return np.full(len(OCTAVE_BANDS), float(c[0]))

    return np.interp(OCTAVE_BANDS, f, c)


def normalise_action(action: str, description: str = "") -> str:
    a = (action or "").strip().lower().replace(" ", "_").replace("-", "_")
    if a in ACTIONS:
        return a

    for name, kws in ACTION_KEYWORDS:
        if any(k in a.replace("_", " ") for k in kws):
            return name

    d = (description or "").lower()
    generic = ("walking", "standing")
    for name, kws in [e for e in ACTION_KEYWORDS if e[0] not in generic] + \
                     [e for e in ACTION_KEYWORDS if e[0] in generic]:
        if any(k in d for k in kws):
            return name
    return "standing"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def outline_polygon(cells):
    cells = set(cells)
    nxt = {}
    for x, y in cells:
        for (dx, dy), edge in (((0, -1), ((x, y), (x + 1, y))), ((1, 0), ((x + 1, y), (x + 1, y + 1))),
                               ((0, 1), ((x + 1, y + 1), (x, y + 1))), ((-1, 0), ((x, y + 1), (x, y)))):
            if (x + dx, y + dy) not in cells:
                if edge[0] in nxt:
                    raise ValueError("the room touches itself at a corner; draw it as one simple shape")
                nxt[edge[0]] = edge[1]
    start = min(nxt)
    loop, p = [start], nxt[start]
    while p != start:
        loop.append(p)
        p = nxt[p]
    if len(loop) != len(nxt):
        raise ValueError("the room must be one connected area without enclosed wall islands")
    return loop


def spatial_map_from_ascii(text: str, scale_m=SCALE_M, ceiling_m=CEILING_M):
    lines = [l.rstrip("\n") for l in text.splitlines() if l.strip()]
    if all(len(l) > 1 and set(l[1::2]) <= {" "} for l in lines):
        lines = [l[0::2] for l in lines]
    width, height = max(len(l) for l in lines), len(lines)
    grid = [list(l.ljust(width).replace(".", " ")) for l in lines]
    inside = {(x, y) for y, r in enumerate(grid) for x, c in enumerate(r) if c not in ("W", " ")}
    if not inside:
        raise ValueError("the map contains no floor tiles")
    if any(x in (0, width - 1) or y in (0, height - 1) for x, y in inside):
        raise ValueError("the map needs a ring of W tiles around the floor")
    corners, problems = normalise_polygon(outline_polygon(inside), width, height)
    problems = [p for p in problems if "smaller than" not in p]
    if problems:
        raise ValueError("; ".join(problems))

    furniture = {}
    for char, key in FURN_CHARS.items():
        cells = [(x, y) for y, r in enumerate(grid) for x, c in enumerate(r) if c == char]
        if not cells:
            continue
        x0, y0 = min(c[0] for c in cells), min(c[1] for c in cells)
        w, h = max(c[0] for c in cells) - x0 + 1, max(c[1] for c in cells) - y0 + 1
        if w * h != len(cells):
            raise ValueError(f"{key}: draw each furniture item once, as a filled rectangle")
        furniture[key] = {"top_left": [x0, y0], "size": [w, h], "center_pos": [x0 + w / 2, y0 + h / 2],
                          "label": FURNITURE[key]["label"], "material": FURNITURE[key]["material"],
                          "height_m": FURNITURE[key]["height"]}
    find = lambda ch: next(([x, y] for y, r in enumerate(grid) for x, c in enumerate(r) if c == ch), None)
    return {"grid": {"width": width, "height": height, "scale_m": scale_m, "ceiling_m": ceiling_m},
            "room_corners": corners, "floor_area_m2": abs(polygon_area(corners)) * scale_m ** 2,
            "furniture_metadata": furniture, "mic_pos": find("M"), "agent_start": find("A")}

import socket
import threading
import sys
import os
import random
import json
import time
from collections import defaultdict, deque

# --- 1. SETUP SERVER ---
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('127.0.0.1', 5005))
server.listen(1)

os.system('cls' if os.name == 'nt' else 'clear')
print("--- ISAAC AI: CONNECTION ESTABLISHMENT ---")
print("Waiting for Isaac (Lua) to connect...")

conn, addr = server.accept()
print(f"SUCCESS: Link established with Isaac at {addr}!")
print("Starting Live Data Stream...\n")

# --- 2. Q-LEARNING CONFIGURATION ---
# The AI can choose to move OR shoot in any direction
COMBAT_ACTIONS = ["MOVE:UP", "MOVE:DOWN", "MOVE:LEFT", "MOVE:RIGHT", "MOVE:STAY", "SHOOT"]
# Navigation uses only movement actions (no SHOOT), but keeps the same table schema.
MOVE_ACTIONS = COMBAT_ACTIONS[:5]

# Lists Enemy names (does not include ALL names, only some)
ENEMY_NAMES = {"10.0": "Gaper", "13.0": "Fly", "85.0": "Monstro"}
# Door slot mapping used to avoid immediately going back through the door we came from.
OPPOSITE_DOOR = {0: 2, 1: 3, 2: 0, 3: 1, 4: 6, 5: 7, 6: 4, 7: 5}
MOVE_TO_VEC = {
    "MOVE:UP": (0, -1),
    "MOVE:DOWN": (0, 1),
    "MOVE:LEFT": (-1, 0),
    "MOVE:RIGHT": (1, 0),
    "MOVE:STAY": (0, 0),
}

# Pickup variants differ by item kind; bias toward notable rewards and keys/bombs over low-value pickups.
PICKUP_VARIANT_PRIORITY = {
    100: 8.0,  # Collectible pedestal
    300: 6.0,  # Chest-like rewards (approx)
    30: 5.0,   # Key-like
    20: 4.5,   # Bomb-like
    10: 4.0,   # Coin-like
    40: 3.5,   # Heart-like
}
CHEST_LIKE_VARIANTS = {50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 300}

# Learning Hyperparameters
# Set TRAIN_FAST=True to squeeze more learning into shorter play sessions.
# It raises alpha/epsilon so each sample counts more, and removes the tick delay.
TRAIN_FAST = True

alpha          = 0.2 if TRAIN_FAST else 0.1   # Learning Rate
gamma          = 0.9                           # Discount Factor
epsilon_combat = 0.25 if TRAIN_FAST else 0.1  # Exploration rate
AUTOSAVE_INTERVAL = 120  # seconds between automatic brain saves


# Initialize or Load Q-table
q_table = defaultdict(lambda: [0.0] * len(COMBAT_ACTIONS))

def save_brain():
    """Saves the learned experience to a JSON file."""
    with open("isaac_brain.json", "w") as f:
        json.dump(dict(q_table), f)
    print("\n[SYSTEM] Brain saved to isaac_brain.json")

def load_brain():
    global q_table
    if os.path.exists("isaac_brain.json"):
        with open("isaac_brain.json", "r") as f:
            data = json.load(f)
            q_table = defaultdict(lambda: [0.0] * len(COMBAT_ACTIONS), data)
        print("[SYSTEM] Previous brain data loaded!")


def choose_nav_target(p_pos, doors, avoid_slot=None):
    """Selects an open door, avoiding immediate backtracking when possible."""
    open_doors = [d for d in doors if d["status"] == "OPEN"]
    if not open_doors:
        return None

    if avoid_slot is not None:
        filtered = [d for d in open_doors if d["slot"] != avoid_slot]
        if filtered:
            open_doors = filtered

    return min(open_doors, key=lambda d: (d["x"] - p_pos[0]) ** 2 + (d["y"] - p_pos[1]) ** 2)


def move_toward_target(p_pos, target, stop_radius=6):
    """Converts a target point into a movement command."""
    if target is None:
        return "MOVE:STAY"

    dx = target["x"] - p_pos[0]
    dy = target["y"] - p_pos[1]

    # Stop if we are close enough to the door (prevents vibrating)
    if abs(dx) < stop_radius and abs(dy) < stop_radius:
        return "MOVE:STAY"

    # Walk in the direction of the biggest gap
    if abs(dx) > abs(dy):
        return "MOVE:RIGHT" if dx > 0 else "MOVE:LEFT"
    return "MOVE:DOWN" if dy > 0 else "MOVE:UP"


def astar(start_idx, goal_idx, grid_w, grid_size, blocked, hazards=None):
    """A* on Isaac's room grid indices, returns a list of indices from start to goal."""
    if start_idx is None or goal_idx is None:
        return None
    if start_idx == goal_idx:
        return [start_idx]
    if grid_w <= 0 or grid_size <= 0:
        return None

    # Start/goal are always treated as walkable so transient collision noise does not brick pathing.
    blocked = set(blocked)
    hazards = set(hazards or [])
    blocked.discard(start_idx)
    blocked.discard(goal_idx)
    hazards.discard(start_idx)

    def valid(idx):
        return 0 <= idx < grid_size and idx not in blocked

    def to_xy(idx):
        return idx % grid_w, idx // grid_w

    def h(a, b):
        ax, ay = to_xy(a)
        bx, by = to_xy(b)
        return abs(ax - bx) + abs(ay - by)

    def neighbors(idx):
        x, y = to_xy(idx)
        cand = []
        if x > 0:
            cand.append(idx - 1)
        if x < grid_w - 1:
            cand.append(idx + 1)
        if y > 0:
            cand.append(idx - grid_w)
        if idx + grid_w < grid_size:
            cand.append(idx + grid_w)
        return [n for n in cand if valid(n)]

    open_set = {start_idx}
    came_from = {}
    g_score = {start_idx: 0}
    f_score = {start_idx: h(start_idx, goal_idx)}

    while open_set:
        current = min(open_set, key=lambda n: f_score.get(n, float("inf")))
        if current == goal_idx:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        open_set.remove(current)
        current_g = g_score.get(current, float("inf"))
        for n in neighbors(current):
            # Crossing hazard tiles is possible but strongly discouraged.
            step_cost = 1 + (8 if n in hazards else 0)
            tentative_g = current_g + step_cost
            if tentative_g < g_score.get(n, float("inf")):
                came_from[n] = current
                g_score[n] = tentative_g
                f_score[n] = tentative_g + h(n, goal_idx)
                open_set.add(n)

    return None


def choose_door_with_novelty(p_pos, doors, room_key, door_visit_counts, avoid_slot=None):
    """Prefer open doors used less often in this room, then nearest by distance."""
    open_doors = [d for d in doors if d["status"] == "OPEN"]
    if not open_doors:
        return None

    if avoid_slot is not None:
        filtered = [d for d in open_doors if d["slot"] != avoid_slot]
        if filtered:
            open_doors = filtered

    def score(door):
        used = door_visit_counts[(room_key, door["slot"])]
        dist2 = (door["x"] - p_pos[0]) ** 2 + (door["y"] - p_pos[1]) ** 2
        return (used, dist2)

    return min(open_doors, key=score)


def move_from_path_step(cur_idx, next_idx, grid_w):
    """Converts one grid step into a MOVE command."""
    delta = next_idx - cur_idx
    if delta == 1:
        return "MOVE:RIGHT"
    if delta == -1:
        return "MOVE:LEFT"
    if delta == grid_w:
        return "MOVE:DOWN"
    if delta == -grid_w:
        return "MOVE:UP"
    return "MOVE:STAY"


def move_away_from_edges(p_grid, grid_w, grid_size):
    """Returns a move that nudges the player away from room edges/corners."""
    if p_grid is None or grid_w <= 0 or grid_size <= 0:
        return None
    x = p_grid % grid_w
    y = p_grid // grid_w
    h = max(1, grid_size // grid_w)

    if x <= 1:
        return "MOVE:RIGHT"
    if x >= grid_w - 2:
        return "MOVE:LEFT"
    if y <= 1:
        return "MOVE:DOWN"
    if y >= h - 2:
        return "MOVE:UP"
    return None


def is_in_corner_zone(p_grid, grid_w, grid_size, band=4):
    """True when position is within a broad corner region, not only the edge tile."""
    if p_grid is None or grid_w <= 0 or grid_size <= 0:
        return False
    x = p_grid % grid_w
    y = p_grid // grid_w
    h = max(1, grid_size // grid_w)

    left = x <= band
    right = x >= grid_w - 1 - band
    top = y <= band
    bottom = y >= h - 1 - band
    return (left and top) or (left and bottom) or (right and top) or (right and bottom)


def choose_engage_move(p_pos, p_grid, enemy, grid_w, grid_size, blocked, hazards):
    """Aggressive reposition move used when combat stalls."""
    enemy_grid = enemy.get("grid")
    if p_grid is not None and enemy_grid is not None and grid_w > 0 and grid_size > 0:
        path = astar(p_grid, enemy_grid, grid_w, grid_size, blocked, hazards)
        if path and len(path) > 1:
            return move_from_path_step(path[0], path[1], grid_w)

    return move_toward_target(p_pos, {"x": enemy["x"], "y": enemy["y"]}, stop_radius=8)


def has_grid_line_blocker(a_idx, b_idx, grid_w, grid_size, blocked):
    """Returns True when a straight grid line between two indices crosses blocked cells."""
    if a_idx is None or b_idx is None or grid_w <= 0 or grid_size <= 0:
        return False

    ax, ay = a_idx % grid_w, a_idx // grid_w
    bx, by = b_idx % grid_w, b_idx // grid_w
    steps = max(abs(bx - ax), abs(by - ay), 1)
    for i in range(1, steps):
        t = i / float(steps)
        x = int(round(ax + (bx - ax) * t))
        y = int(round(ay + (by - ay) * t))
        idx = y * grid_w + x
        if 0 <= idx < grid_size and idx in blocked:
            return True
    return False


def move_result_position(p_pos, move_cmd, step=28.0):
    """Predicts a short-horizon position after one move command."""
    vx, vy = MOVE_TO_VEC.get(move_cmd, (0, 0))
    return p_pos[0] + vx * step, p_pos[1] + vy * step


def score_move_safety(move_cmd, p_pos, enemies, projectiles, p_grid=None, grid_w=0, grid_size=0, hazards=None):
    """Higher is safer. Penalizes moving toward enemies/projectile trajectories."""
    nx, ny = move_result_position(p_pos, move_cmd)

    score = 0.0
    for e in enemies:
        ex = e["x"] - nx
        ey = e["y"] - ny
        dist2 = ex * ex + ey * ey
        score += min(dist2, 50000.0) / 50000.0

        # Extra penalty for being near fast enemies.
        speed = abs(e.get("vx", 0.0)) + abs(e.get("vy", 0.0))
        if speed > 0.8 and dist2 < 140 * 140:
            score -= 2.0

    for t in projectiles:
        tx = t["x"] - nx
        ty = t["y"] - ny
        dist2 = tx * tx + ty * ty
        if dist2 < 95 * 95:
            score -= 3.0

        # Penalize stepping into the projectile's near-future position.
        fx = t["x"] + t.get("vx", 0.0) * 5
        fy = t["y"] + t.get("vy", 0.0) * 5
        fdx = fx - nx
        fdy = fy - ny
        if (fdx * fdx + fdy * fdy) < 85 * 85:
            score -= 3.5

    if hazards is not None and p_grid is not None and grid_w > 0 and grid_size > 0:
        dx, dy = MOVE_TO_VEC.get(move_cmd, (0, 0))
        next_idx = p_grid + dx + dy * grid_w
        if 0 <= next_idx < grid_size and next_idx in hazards:
            score -= 6.0

    if move_cmd == "MOVE:STAY":
        score -= 0.3
    return score


def choose_safe_move(preferred_move, p_pos, enemies, projectiles, p_grid=None, grid_w=0, grid_size=0, hazards=None):
    """Keeps intent when safe, otherwise picks a safer direction."""
    candidates = ["MOVE:UP", "MOVE:DOWN", "MOVE:LEFT", "MOVE:RIGHT", "MOVE:STAY"]
    base = score_move_safety(preferred_move, p_pos, enemies, projectiles, p_grid, grid_w, grid_size, hazards)

    best_move = preferred_move
    best_score = base
    for c in candidates:
        s = score_move_safety(c, p_pos, enemies, projectiles, p_grid, grid_w, grid_size, hazards)
        if c == preferred_move:
            s += 0.25
        if s > best_score:
            best_move = c
            best_score = s

    return best_move


def choose_item_target(items, p_pos):
    """Picks a loot target with a simple value-vs-distance tradeoff."""
    if not items:
        return None

    def utility(item):
        priority = PICKUP_VARIANT_PRIORITY.get(item["variant"], 2.5)
        dist2 = (item["x"] - p_pos[0]) ** 2 + (item["y"] - p_pos[1]) ** 2
        return priority * 10000.0 - dist2

    return max(items, key=utility)


def is_affordable_item(item, coins):
    """Returns True when item has no price or can be purchased with current coins."""
    price = item.get("price", 0)
    if price <= 0:
        return True
    return coins >= price


def should_consider_item(item, coins, red_hearts, max_red_hearts):
    """Filters pickups that are currently non-actionable or harmful to policy flow."""
    if not is_affordable_item(item, coins):
        return False

    variant = item.get("variant", -1)

    # Ignore chest-like physics objects to avoid pathing loops on pushable/non-pickup entities.
    if variant in CHEST_LIKE_VARIANTS:
        return False

    # Skip red-heart pickups when red health is already full.
    if variant == 40 and max_red_hearts > 0 and red_hearts >= max_red_hearts:
        return False

    return True


def choose_dodge_move(p_pos, projectiles):
    """Dodges the nearest threatening projectile using a perpendicular strafe."""
    if not projectiles:
        return None

    nearest = min(projectiles, key=lambda t: (t["x"] - p_pos[0]) ** 2 + (t["y"] - p_pos[1]) ** 2)
    dx = nearest["x"] - p_pos[0]
    dy = nearest["y"] - p_pos[1]
    dist2 = dx * dx + dy * dy
    if dist2 > 120 * 120:
        return None

    vx = nearest["vx"]
    vy = nearest["vy"]
    speed2 = vx * vx + vy * vy

    # If projectile has little velocity info, move away directly from its position.
    if speed2 < 0.05:
        if abs(dx) > abs(dy):
            return "MOVE:LEFT" if dx > 0 else "MOVE:RIGHT"
        return "MOVE:UP" if dy > 0 else "MOVE:DOWN"

    # Perpendicular dodge to projectile velocity vector.
    side_dx = -vy
    side_dy = vx
    if abs(side_dx) > abs(side_dy):
        return "MOVE:RIGHT" if side_dx > 0 else "MOVE:LEFT"
    return "MOVE:DOWN" if side_dy > 0 else "MOVE:UP"


def choose_unstuck_move(p_grid, grid_w, grid_size, blocked):
    """When stuck, step toward room center (A*) or choose a random open local move."""
    if p_grid is not None and grid_w > 0 and grid_size > 0:
        center_idx = grid_size // 2
        path = astar(p_grid, center_idx, grid_w, grid_size, blocked)
        if path and len(path) > 1:
            return move_from_path_step(path[0], path[1], grid_w)

        options = []
        for cmd, (dx, dy) in MOVE_TO_VEC.items():
            if cmd == "MOVE:STAY":
                continue
            n = p_grid + dx + dy * grid_w
            if 0 <= n < grid_size and n not in blocked:
                options.append(cmd)
        if options:
            return random.choice(options)

    return random.choice(["MOVE:UP", "MOVE:DOWN", "MOVE:LEFT", "MOVE:RIGHT"])


def replace_move_in_combined_command(cmd, new_move):
    """Replaces the MOVE segment in a combined command while preserving SHOOT segment."""
    if ";" not in cmd:
        return new_move
    parts = cmd.split(";")
    out = []
    moved = False
    for p in parts:
        if p.startswith("MOVE:") and not moved:
            out.append(new_move)
            moved = True
        else:
            out.append(p)
    if not moved:
        out.insert(0, new_move)
    return ";".join(out)


def choose_combat_move(dx, dy, enemy_speed, p_pos, projectiles):
    """Threat-aware movement: dodge projectiles, kite movers, chase stationary targets."""
    dodge_cmd = choose_dodge_move(p_pos, projectiles)
    if dodge_cmd:
        return dodge_cmd

    dist2 = dx * dx + dy * dy

    # If target is mostly stationary and far away, chase aggressively to avoid passive camping.
    if enemy_speed < 0.35 and dist2 > 95 * 95:
        if abs(dx) > abs(dy):
            return "MOVE:RIGHT" if dx > 0 else "MOVE:LEFT"
        return "MOVE:DOWN" if dy > 0 else "MOVE:UP"

    # Fast movers nearby: create space.
    if enemy_speed > 0.9 and dist2 < 150 * 150:
        if abs(dx) > abs(dy):
            return "MOVE:LEFT" if dx > 0 else "MOVE:RIGHT"
        return "MOVE:UP" if dy > 0 else "MOVE:DOWN"

    # Too close: back away from enemy.
    if dist2 < 75 * 75:
        if abs(dx) > abs(dy):
            return "MOVE:LEFT" if dx > 0 else "MOVE:RIGHT"
        return "MOVE:UP" if dy > 0 else "MOVE:DOWN"

    # Too far: close distance so shots are more likely to connect.
    if dist2 > 170 * 170:
        if abs(dx) > abs(dy):
            return "MOVE:RIGHT" if dx > 0 else "MOVE:LEFT"
        return "MOVE:DOWN" if dy > 0 else "MOVE:UP"

    # Mid-range: strafe perpendicular to enemy direction.
    if abs(dx) > abs(dy):
        return "MOVE:DOWN" if dy >= 0 else "MOVE:UP"
    return "MOVE:RIGHT" if dx >= 0 else "MOVE:LEFT"


# Load persisted Q-values before starting the real-time loop.
load_brain()

# --- 3. THE BRAIN (DATA PROCESSING) ---
def listen_to_isaac(conn):
    # Memory variables for the Markov Process
    # These track transition context needed for reward shaping and anti-loop navigation.
    last_state = "BOOT"
    last_action_idx = COMBAT_ACTIONS.index("MOVE:STAY")
    last_hp = None
    last_autosave_time = time.time()
    last_room_key = None
    last_exit_slot = None
    door_visit_counts = defaultdict(int)
    nav_target_slot = None
    recent_grids = deque(maxlen=14)
    unreachable_item_targets = {}
    focused_item_key = None
    focused_item_ticks = 0
    recent_combat_grids = deque(maxlen=20)
    corner_camp_ticks = 0
    recent_enemy_dist2 = deque(maxlen=14)

    while True:
        try:
            raw_data = conn.recv(4096).decode('utf-8').strip()
            if not raw_data:
                continue

            # Get the most recent packet
            latest_line = raw_data.split('\n')[-1]
            parts = latest_line.split("|")
            
            # P:X,Y | H:HP | R:STAGE:SAFE_GRID_INDEX
            # This is the packet built in Lua, split by "|" components.
            p_info = parts[0][2:].split(",")
            p_pos = [float(p_info[0]), float(p_info[1])]
            p_grid = int(p_info[2]) if len(p_info) >= 3 else None
            hp = int(parts[1][2:])
            red_hearts = 0
            max_red_hearts = 0
            coins = 0
            room_key = "UNKNOWN"
            grid_w = 0
            grid_size = 0
            blocked = set()
            hazards = set()
            
            # SENSING: Enemies & Doors
            enemies = []
            items = []
            projectiles = []
            doors = []
            for p in parts:
                if p.startswith("E:"):
                    e_parts = p.split(":")
                    pos = e_parts[2].split(",")
                    vel = [0.0, 0.0]
                    if len(e_parts) >= 4:
                        vel_raw = e_parts[3].split(",")
                        if len(vel_raw) >= 2:
                            vel = [float(vel_raw[0]), float(vel_raw[1])]
                    e_grid = None
                    if len(e_parts) >= 5:
                        try:
                            e_grid = int(e_parts[4])
                        except ValueError:
                            e_grid = None
                    enemies.append({
                        "id": e_parts[1],
                        "x": float(pos[0]),
                        "y": float(pos[1]),
                        "vx": vel[0],
                        "vy": vel[1],
                        "grid": e_grid,
                    })
                if p.startswith("I:"):
                    i_parts = p.split(":")
                    i_pos = i_parts[2].split(",")
                    i_grid = None
                    if len(i_parts) >= 4:
                        try:
                            i_grid = int(i_parts[3])
                        except ValueError:
                            i_grid = None
                    i_type, i_variant = i_parts[1].split(".")
                    price = 0
                    if len(i_parts) >= 5:
                        try:
                            price = int(i_parts[4])
                        except ValueError:
                            price = 0
                    items.append({
                        "type": int(i_type),
                        "variant": int(i_variant),
                        "x": float(i_pos[0]),
                        "y": float(i_pos[1]),
                        "grid": i_grid,
                        "price": price,
                    })
                if p.startswith("T:"):
                    t_parts = p.split(":")
                    t_pos = t_parts[1].split(",")
                    t_vel = [0.0, 0.0]
                    if len(t_parts) >= 3:
                        t_vel_raw = t_parts[2].split(",")
                        if len(t_vel_raw) >= 2:
                            t_vel = [float(t_vel_raw[0]), float(t_vel_raw[1])]
                    t_grid = None
                    if len(t_parts) >= 4:
                        try:
                            t_grid = int(t_parts[3])
                        except ValueError:
                            t_grid = None
                    projectiles.append({
                        "x": float(t_pos[0]),
                        "y": float(t_pos[1]),
                        "vx": t_vel[0],
                        "vy": t_vel[1],
                        "grid": t_grid,
                    })
                if p.startswith("D:"):
                    # Split the door string so we can read status
                    d_info = p.split(":") 
                    # Capture status AND the X,Y coordinates
                    coords = d_info[3].split(",")
                    door_grid = None
                    if len(d_info) >= 5:
                        try:
                            door_grid = int(d_info[4])
                        except ValueError:
                            door_grid = None
                    doors.append({
                        "slot": int(d_info[1]),
                        "status": d_info[2], 
                        "x": float(coords[0]), 
                        "y": float(coords[1]),
                        "grid": door_grid,
                    })
                if p.startswith("R:"):
                    r_info = p.split(":")
                    if len(r_info) >= 3:
                        room_key = f"{r_info[1]}:{r_info[2]}"
                if p.startswith("C:"):
                    try:
                        coins = int(p[2:])
                    except ValueError:
                        coins = 0
                if p.startswith("V:"):
                    v_info = p.split(":")
                    if len(v_info) >= 3:
                        try:
                            red_hearts = int(v_info[1])
                            max_red_hearts = int(v_info[2])
                        except ValueError:
                            red_hearts = 0
                            max_red_hearts = 0
                if p.startswith("G:"):
                    g_info = p.split(":", 3)
                    if len(g_info) >= 3:
                        grid_w = int(g_info[1])
                        grid_size = int(g_info[2])
                    if len(g_info) == 4 and g_info[3]:
                        blocked = {int(x) for x in g_info[3].split(",") if x.strip() != ""}
                if p.startswith("Z:"):
                    z_info = p.split(":", 1)
                    if len(z_info) == 2 and z_info[1]:
                        hazards = {int(x) for x in z_info[1].split(",") if x.strip() != ""}

            # --- STATE & AIM MATH ---
            # On room transitions, avoid the opposite of the previous exit door when possible.
            room_changed = last_room_key is not None and room_key != last_room_key
            avoid_slot = None
            if room_changed and last_exit_slot is not None:
                avoid_slot = OPPOSITE_DOOR.get(last_exit_slot)
                if last_room_key is not None:
                    door_visit_counts[(last_room_key, last_exit_slot)] += 1
            if room_changed:
                nav_target_slot = None
                unreachable_item_targets = {}
                focused_item_key = None
                focused_item_ticks = 0
                recent_combat_grids.clear()
                corner_camp_ticks = 0
                recent_enemy_dist2.clear()

            if p_grid is not None:
                recent_grids.append(p_grid)

            target_dir = "NONE"
            if enemies:
                def enemy_threat_score(e):
                    dist2 = (e['x'] - p_pos[0]) ** 2 + (e['y'] - p_pos[1]) ** 2
                    speed = abs(e['vx']) + abs(e['vy'])
                    return dist2 * (0.6 if speed > 0.7 else 1.0)

                closest = min(enemies, key=enemy_threat_score)
                dx, dy = closest['x'] - p_pos[0], closest['y'] - p_pos[1]
                dist2_to_enemy = dx * dx + dy * dy
                enemy_speed = abs(closest['vx']) + abs(closest['vy'])
                enemy_grid = closest.get('grid')

                if p_grid is not None:
                    recent_combat_grids.append(p_grid)
                # Stalled if bouncing between ≤4 unique tiles for the full window (catches jitter).
                combat_stalled = (
                    len(recent_combat_grids) == recent_combat_grids.maxlen
                    and len(set(recent_combat_grids)) <= 4
                )

                rel_x = "RIGHT" if dx > 0 else "LEFT"
                rel_y = "DOWN" if dy > 0 else "UP"
                name = ENEMY_NAMES.get(closest['id'], f"ID_{closest['id']}")
                
                current_state = f"COMBAT_{name}_{rel_x}_{rel_y}"
                target_dir = rel_x if abs(dx) > abs(dy) else rel_y

                # --- DECIDE NEXT ACTION (epsilon-greedy) ---
                if random.random() < epsilon_combat:
                    action_idx = random.randint(0, len(COMBAT_ACTIONS) - 1)
                else:
                    action_idx = q_table[current_state].index(max(q_table[current_state]))

                # --- EXECUTE ACTION ---
                # Always keep pressure by shooting in combat while moving safely.
                chosen_move = COMBAT_ACTIONS[action_idx]
                if chosen_move.startswith("MOVE:"):
                    move_cmd = chosen_move
                else:
                    move_cmd = choose_combat_move(dx, dy, enemy_speed, p_pos, projectiles)

                # If enemy is behind blocking geometry, reposition via A* instead of tunnel-shooting walls.
                los_blocked = has_grid_line_blocker(p_grid, enemy_grid, grid_w, grid_size, blocked)
                recent_enemy_dist2.append(dist2_to_enemy)
                distance_not_improving = (
                    len(recent_enemy_dist2) == recent_enemy_dist2.maxlen
                    and recent_enemy_dist2[-1] >= (recent_enemy_dist2[0] * 0.92)
                )
                in_corner_zone = is_in_corner_zone(p_grid, grid_w, grid_size)
                # Increment whenever cornered OR distance is stuck — no AND gate needed.
                if in_corner_zone or distance_not_improving:
                    corner_camp_ticks += 1
                else:
                    corner_camp_ticks = max(0, corner_camp_ticks - 1)

                if los_blocked and p_grid is not None and enemy_grid is not None and grid_w > 0 and grid_size > 0:
                    chase_path = astar(p_grid, enemy_grid, grid_w, grid_size, blocked, hazards)
                    if chase_path and len(chase_path) > 1:
                        move_cmd = move_from_path_step(chase_path[0], chase_path[1], grid_w)

                # Only apply safety filtering when NOT overriding for a stall/corner escape.
                if not (combat_stalled or corner_camp_ticks > 10):
                    move_cmd = choose_safe_move(move_cmd, p_pos, enemies, projectiles, p_grid, grid_w, grid_size, hazards)

                # Force engagement when stalled in a tile cluster or camping a corner too long.
                if combat_stalled or corner_camp_ticks > 10:
                    edge_escape = move_away_from_edges(p_grid, grid_w, grid_size)
                    engage_move = edge_escape or choose_engage_move(
                        p_pos, p_grid, closest, grid_w, grid_size, blocked, hazards
                    )
                    move_cmd = engage_move

                shoot_cmd = f"SHOOT:{target_dir}" if target_dir != "NONE" else "SHOOT:NONE"
                final_cmd = f"{move_cmd};{shoot_cmd}"

            else:
                recent_combat_grids.clear()
                corner_camp_ticks = 0
                recent_enemy_dist2.clear()
                # --- ROOM IS CLEAR: navigate to a useful door ---
                # Encode room context in state so navigation can also be learned over time.
                open_doors = [d for d in doors if d["status"] == "OPEN"]
                current_state = f"CLEAR_{room_key}_OD{len(open_doors)}"

                # Prefer collecting room pickups before taking a door.
                filtered_items = []
                for it in items:
                    if not should_consider_item(it, coins, red_hearts, max_red_hearts):
                        continue
                    key = (room_key, it.get("grid"), it.get("variant"))
                    blocked_until = unreachable_item_targets.get(key, 0)
                    if blocked_until <= time.time():
                        filtered_items.append(it)
                item_target = choose_item_target(filtered_items, p_pos)
                if item_target:
                    item_key = (room_key, item_target.get("grid"), item_target.get("variant"))
                    item_dist2 = (item_target["x"] - p_pos[0]) ** 2 + (item_target["y"] - p_pos[1]) ** 2
                    if focused_item_key == item_key and item_dist2 < 42 * 42:
                        focused_item_ticks += 1
                    else:
                        focused_item_key = item_key
                        focused_item_ticks = 0

                    # Typical store lock: we can stand on it but cannot pick it up.
                    if focused_item_ticks > 25:
                        unreachable_item_targets[item_key] = time.time() + 45.0
                        item_target = None
                        focused_item_key = None
                        focused_item_ticks = 0

                # Keep a stable target door during a room to avoid target flipping and circles.
                door_target = None
                if nav_target_slot is not None:
                    door_target = next((d for d in open_doors if d["slot"] == nav_target_slot), None)
                if door_target is None:
                    door_target = choose_door_with_novelty(p_pos, doors, room_key, door_visit_counts, avoid_slot)
                    if door_target:
                        nav_target_slot = door_target["slot"]

                target = item_target if item_target else door_target
                heuristic_cmd = move_toward_target(p_pos, target, stop_radius=4 if item_target else 3)

                path_cmd = None
                target_grid = target.get("grid") if target else None
                if (
                    target
                    and p_grid is not None
                    and target_grid is not None
                    and grid_w > 0
                    and grid_size > 0
                ):
                    path = astar(p_grid, target_grid, grid_w, grid_size, blocked, hazards)
                    if path and len(path) > 1:
                        # If route immediately enters a hazard tile, treat target as temporarily not worth it.
                        if len(path) > 1 and path[1] in hazards and item_target and target is item_target:
                            it_key = (room_key, item_target.get("grid"), item_target.get("variant"))
                            unreachable_item_targets[it_key] = time.time() + 20.0
                            item_target = None
                            target = door_target
                            target_grid = target.get("grid") if target else None
                            if target_grid is not None:
                                path = astar(p_grid, target_grid, grid_w, grid_size, blocked, hazards)
                        if path and len(path) > 1:
                            path_cmd = move_from_path_step(path[0], path[1], grid_w)
                    elif path and len(path) == 1:
                        # Even on the target tile, keep nudging toward center to complete pickup/transition.
                        path_cmd = move_toward_target(p_pos, target, stop_radius=0)
                    elif item_target and target is item_target:
                        # If an item/chest is unreachable, skip it for a while so we do not deadlock.
                        it_key = (room_key, item_target.get("grid"), item_target.get("variant"))
                        unreachable_item_targets[it_key] = time.time() + 8.0
                        item_target = None
                        target = door_target
                        heuristic_cmd = move_toward_target(p_pos, target, stop_radius=3)

                # In clear rooms, prioritize deterministic path-following to prevent circling.
                final_cmd = path_cmd or heuristic_cmd
                if target and final_cmd == "MOVE:STAY":
                    final_cmd = move_toward_target(p_pos, target, stop_radius=0)
                final_cmd = choose_safe_move(final_cmd, p_pos, enemies, projectiles, p_grid, grid_w, grid_size, hazards)
                action_idx = COMBAT_ACTIONS.index(final_cmd)

                if door_target and target is door_target and final_cmd == "MOVE:STAY":
                    # We likely touched this door; remember it so next room can avoid immediate reversal.
                    last_exit_slot = door_target["slot"]
                elif door_target and target is door_target and final_cmd.startswith("MOVE:"):
                    # Keep exit intent bound to the active target while we approach it.
                    last_exit_slot = door_target["slot"]

            # If movement command keeps us in nearly the same tiles for too long, force an escape action.
            grid_stuck = len(recent_grids) == recent_grids.maxlen and len(set(recent_grids)) <= 2
            if grid_stuck:
                unstuck_move = choose_unstuck_move(p_grid, grid_w, grid_size, blocked)
                final_cmd = replace_move_in_combined_command(final_cmd, unstuck_move)
                if ";" not in final_cmd:
                    action_idx = COMBAT_ACTIONS.index(unstuck_move)

            # --- REWARD CALCULATION ---
            # Lightweight shaping: penalize damage, reward room progress, and mild combat survival.
            reward = 0
            if last_hp is not None and hp < last_hp:
                reward -= 100
            if last_state.startswith("COMBAT_") and current_state.startswith("CLEAR_"):
                reward += 75
            if room_changed:
                reward += 25
            if enemies and (last_hp is None or hp >= last_hp):
                reward += 1

            # --- Q-LEARNING UPDATE ---
            # Standard tabular Q-learning update:
            # Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
            if last_hp is not None:
                old_q = q_table[last_state][last_action_idx]
                max_future_q = max(q_table[current_state])
                q_table[last_state][last_action_idx] = old_q + alpha * (reward + gamma * max_future_q - old_q)

            conn.sendall(f"{final_cmd}\n".encode('utf-8'))

            # Convert raw coordinates to "Grid" units (approx 40px per tile)
            grid_x = int(p_pos[0] / 40)
            grid_y = int(p_pos[1] / 40)
            open_doors_count = len([d for d in doors if d['status'] == "OPEN"])
            
            # Line 1: Main Stats
            msg = f"\033[H[LIVE DATA] HP: {hp} | Grid: ({grid_x},{grid_y}) | Room: {room_key} | State: {current_state:20}\n"
            
            # Line 2: Room Info
            if not enemies:
                status_msg = "NAVIGATING" if open_doors_count > 0 else "WAITING..."
            else:
                status_msg = "IN COMBAT"

            msg += (
                f"Enemies: {len(enemies):<3} | Shots: {len(projectiles):<3} | "
                f"Items: {len(items):<3} | Open Doors: {open_doors_count:<2} | "
                f"Reward: {reward:<5} | STATUS: {status_msg} | Cmd: {final_cmd:18}"
            )
            
            sys.stdout.write(msg)
            sys.stdout.flush()

            # Update Memory
            last_state, last_action_idx, last_hp = current_state, action_idx, hp
            last_room_key = room_key

            # Periodic autosave so a crash never loses more than AUTOSAVE_INTERVAL seconds.
            now = time.time()
            if now - last_autosave_time >= AUTOSAVE_INTERVAL:
                save_brain()
                last_autosave_time = now

            if not TRAIN_FAST:
                time.sleep(0.05)
                
        except Exception as e:
            continue

# Start the background thread
threading.Thread(target=listen_to_isaac, args=(conn,), daemon=True).start()

# --- 4. THE HANDS (MANUAL OVERRIDE) ---
print("\n" * 3) 
try:
    while True:
        cmd = input(">> ENTER COMMAND (spawn/save/exit): ").strip().lower()
        if cmd == "save":
            save_brain()
        elif cmd == "exit":
            break
        elif cmd:
            conn.sendall((cmd + "\n").encode('utf-8'))
            sys.stdout.write(f"\033[KLast Action Sent: {cmd}\n") 
except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    save_brain()
    conn.close()
    server.close()
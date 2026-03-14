import csv
import json
import random
import socket
import time
from collections import defaultdict, deque
from pathlib import Path

from .agent import DQNAgent
from .config import Config, decode_combat_action
from .features import featurize_state
from .protocol import parse_packet


OPPOSITE_DOOR = {0: 2, 1: 3, 2: 0, 3: 1, 4: 6, 5: 7, 6: 4, 7: 5}
HEART_VARIANT = 10  # PickupVariant.PICKUP_HEART
BUTTON_ROOM_STUCK_RESTART_TICKS = 240
STUCK_POS_EPS = 2.0
COMBAT_STUCK_WINDOW = 24
COMBAT_STUCK_MAX_UNIQUE_GRIDS = 3
COMBAT_STUCK_REPATH_TICKS = 90
CLEAR_STUCK_WINDOW = 32
CLEAR_STUCK_MAX_UNIQUE_GRIDS = 3
CLEAR_STUCK_RESTART_TICKS = 220
MAX_EPISODE_SECONDS = 300
HAZARD_CONTACT_PENALTY = 20.0
ENEMY_COLLISION_DIST = 34
ENEMY_COLLISION_PENALTY = 8.0
ENEMY_NEAR_DIST = 52
ENEMY_NEAR_PENALTY = 1.5


def load_and_increment_run_count(counter_path: Path) -> int:
    counter_path.parent.mkdir(parents=True, exist_ok=True)

    run_count = 0
    if counter_path.exists():
        try:
            with counter_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                run_count = int(data.get("run_count", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            run_count = 0

    run_count += 1
    with counter_path.open("w", encoding="utf-8") as f:
        json.dump({"run_count": run_count}, f)

    return run_count


def move_from_path_step(cur_idx, next_idx, grid_w):
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


def move_toward_xy(src_x, src_y, dst_x, dst_y):
    dx = dst_x - src_x
    dy = dst_y - src_y
    if abs(dx) >= abs(dy):
        return "MOVE:RIGHT" if dx > 0 else "MOVE:LEFT"
    return "MOVE:DOWN" if dy > 0 else "MOVE:UP"


def door_nudge_move(player_x, player_y, door):
    """Choose a stable approach direction near a door to avoid jitter.

    For side doors prefer horizontal movement; for top/bottom doors prefer vertical.
    Falls back to generic vector movement for non-standard slots.
    """
    dx = door.x - player_x
    dy = door.y - player_y
    if door.slot in (1, 3):
        return "MOVE:RIGHT" if dx > 0 else "MOVE:LEFT"
    if door.slot in (0, 2):
        return "MOVE:DOWN" if dy > 0 else "MOVE:UP"
    return move_toward_xy(player_x, player_y, door.x, door.y)


def _next_idx_for_move(cur_idx, move_cmd, grid_w, grid_size):
    if cur_idx is None or grid_w <= 0 or grid_size <= 0:
        return None
    x = cur_idx % grid_w
    y = cur_idx // grid_w
    h = max(1, grid_size // grid_w)
    if move_cmd == "MOVE:LEFT":
        return cur_idx - 1 if x > 0 else None
    if move_cmd == "MOVE:RIGHT":
        return cur_idx + 1 if x < grid_w - 1 else None
    if move_cmd == "MOVE:UP":
        return cur_idx - grid_w if y > 0 else None
    if move_cmd == "MOVE:DOWN":
        return cur_idx + grid_w if y < h - 1 else None
    if move_cmd == "MOVE:STAY":
        return cur_idx
    return None


def sanitize_move_cmd(state, move_cmd, fallback_cmd="MOVE:STAY", allowed_target_idx=None):
    """Prevent movement into blocked/hazard tiles.

    Returns the requested move if valid, otherwise fallback_cmd if valid,
    otherwise a first available cardinal move, otherwise MOVE:STAY.
    """
    if state.player_grid is None or state.grid_w <= 0 or state.grid_size <= 0:
        return move_cmd

    blocked = state.blocked | state.hazards

    def is_valid(cmd):
        if cmd == "MOVE:STAY":
            return True
        nxt = _next_idx_for_move(state.player_grid, cmd, state.grid_w, state.grid_size)
        if nxt is None:
            return False
        if allowed_target_idx is not None and nxt == allowed_target_idx:
            return True
        return nxt not in blocked

    if is_valid(move_cmd):
        return move_cmd
    if is_valid(fallback_cmd):
        return fallback_cmd

    for cmd in ("MOVE:LEFT", "MOVE:RIGHT", "MOVE:UP", "MOVE:DOWN"):
        if is_valid(cmd):
            return cmd
    return "MOVE:STAY"


def random_valid_move(state, allow_stay=False):
    """Pick a random valid cardinal move to break repetitive loops."""
    cmds = ["MOVE:UP", "MOVE:DOWN", "MOVE:LEFT", "MOVE:RIGHT"]
    random.shuffle(cmds)

    if state.player_grid is None or state.grid_w <= 0 or state.grid_size <= 0:
        return random.choice(cmds) if cmds else "MOVE:STAY"

    blocked = state.blocked | state.hazards
    for cmd in cmds:
        nxt = _next_idx_for_move(state.player_grid, cmd, state.grid_w, state.grid_size)
        if nxt is not None and nxt not in blocked:
            return cmd
    return "MOVE:STAY" if allow_stay else random.choice(cmds)


def choose_combat_chase_move(state, nearest_enemy):
    """Returns (move_cmd, shoot_override).

    shoot_override is None normally; a cardinal direction when poop is
    blocking the path and needs to be cleared before we can advance.
    """
    if nearest_enemy is None:
        return "MOVE:STAY", None

    if (
        state.player_grid is not None
        and nearest_enemy.grid is not None
        and state.grid_w > 0
        and state.grid_size > 0
    ):
        nav_blocked = state.blocked | state.hazards

        # Try walking path ignoring poop (poop is destructible).
        poop_passable = nav_blocked - state.poops
        path = astar(state.player_grid, nearest_enemy.grid, state.grid_w, state.grid_size, poop_passable)
        if path and len(path) > 1:
            next_idx = path[1]
            # If the next step is a poop tile, stand still and shoot it instead of
            # walking into it (shooting destroys it faster than bumping).
            if next_idx in state.poops:
                shoot_dir = find_fire_shoot_dir(state.player_grid, nearest_enemy.grid, state.grid_w, state.poops)
                return "MOVE:STAY", shoot_dir
            return move_from_path_step(path[0], path[1], state.grid_w), None

        # Even poop-passable path failed; try full-blocked A* for a normal move.
        path_full = astar(state.player_grid, nearest_enemy.grid, state.grid_w, state.grid_size, nav_blocked)
        if path_full and len(path_full) > 1:
            return move_from_path_step(path_full[0], path_full[1], state.grid_w), None

    return move_toward_xy(state.player_x, state.player_y, nearest_enemy.x, nearest_enemy.y), None


def astar(start_idx, goal_idx, grid_w, grid_size, blocked):
    if start_idx is None or goal_idx is None:
        return None
    if start_idx == goal_idx:
        return [start_idx]
    if grid_w <= 0 or grid_size <= 0:
        return None

    blocked = set(blocked)
    blocked.discard(start_idx)
    blocked.discard(goal_idx)

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
            tentative_g = current_g + 1
            if tentative_g < g_score.get(n, float("inf")):
                came_from[n] = current
                g_score[n] = tentative_g
                f_score[n] = tentative_g + h(n, goal_idx)
                open_set.add(n)

    return None


def find_fire_shoot_dir(player_grid, target_grid, grid_w, hazards):
    """Return shoot direction toward the nearest hazard (fire/spikes) that lies
    between the player and the target tile.  Returns None if there are no
    hazards in the direction of travel."""
    if player_grid is None or target_grid is None or not hazards or grid_w <= 0:
        return None
    px, py = player_grid % grid_w, player_grid // grid_w
    tx, ty = target_grid % grid_w, target_grid // grid_w
    dx, dy = tx - px, ty - py
    if dx == 0 and dy == 0:
        return None

    best_hdxdy = None
    best_dist = float("inf")
    for h in hazards:
        hx, hy = h % grid_w, h // grid_w
        hdx, hdy = hx - px, hy - py
        if hdx == 0 and hdy == 0:
            continue
        # Only consider hazards that are in the general direction of the target.
        if hdx * dx + hdy * dy <= 0:
            continue
        dist = abs(hdx) + abs(hdy)
        if dist < best_dist:
            best_dist = dist
            best_hdxdy = (hdx, hdy)

    if best_hdxdy is None:
        return None
    hdx, hdy = best_hdxdy
    if abs(hdx) >= abs(hdy):
        return "RIGHT" if hdx > 0 else "LEFT"
    return "DOWN" if hdy > 0 else "UP"


def is_cardinal_los_blocked(player_grid, target_grid, grid_w, blocked):
    """True when player and target are aligned and an obstacle lies between.

    This is used to avoid pointlessly shooting into rocks/walls in combat.
    """
    if player_grid is None or target_grid is None or grid_w <= 0:
        return False

    px, py = player_grid % grid_w, player_grid // grid_w
    tx, ty = target_grid % grid_w, target_grid // grid_w

    if px == tx:
        step = grid_w if ty > py else -grid_w
        cur = player_grid + step
        while cur != target_grid:
            if cur in blocked:
                return True
            cur += step
        return False

    if py == ty:
        step = 1 if tx > px else -1
        cur = player_grid + step
        while cur != target_grid:
            if cur in blocked:
                return True
            cur += step
        return False

    return False


def choose_clear_room_move(state, nav_target_slot, avoid_slot=None, unreachable_item_targets=None):
    """Returns (move_cmd, nav_target_slot, shoot_cmd).

    shoot_cmd is 'NONE' unless fire/spikes are blocking the path, in which
    case it holds a cardinal direction to shoot toward the obstruction.
    """
    if unreachable_item_targets is None:
        unreachable_item_targets = {}

    # Include hazards (fire, spikes) in the blocked set so A* avoids them.
    nav_blocked = state.blocked | state.hazards

    # ── 1. Red-heart pickup ────────────────────────────────────────────────
    filtered_items = []
    for it in state.items:
        if it.item_type != 5:
            continue
        if it.variant != HEART_VARIANT:
            continue
        if state.max_red_hearts > 0 and state.red_hearts >= state.max_red_hearts:
            continue
        key = (state.room_key, it.grid, it.variant)
        if unreachable_item_targets.get(key, 0.0) > time.time():
            continue
        filtered_items.append(it)

    item_target = None
    if filtered_items:
        item_target = min(
            filtered_items,
            key=lambda it: (it.x - state.player_x) ** 2 + (it.y - state.player_y) ** 2,
        )

    if item_target is not None:
        if (
            state.player_grid is not None
            and item_target.grid is not None
            and state.grid_w > 0
            and state.grid_size > 0
        ):
            path = astar(state.player_grid, item_target.grid, state.grid_w, state.grid_size, nav_blocked)
            if path and len(path) > 1:
                return move_from_path_step(path[0], path[1], state.grid_w), nav_target_slot, "NONE"
            if path and len(path) == 1:
                return "MOVE:STAY", nav_target_slot, "NONE"
            # Unreachable (fire or rocks in the way); skip temporarily.
            item_key = (state.room_key, item_target.grid, item_target.variant)
            unreachable_item_targets[item_key] = time.time() + 20.0
            item_target = None

    if item_target is not None:
        dx = item_target.x - state.player_x
        dy = item_target.y - state.player_y
        if abs(dx) < 10 and abs(dy) < 10:
            return "MOVE:STAY", nav_target_slot, "NONE"
        if abs(dx) > abs(dy):
            return ("MOVE:RIGHT" if dx > 0 else "MOVE:LEFT"), nav_target_slot, "NONE"
        return ("MOVE:DOWN" if dy > 0 else "MOVE:UP"), nav_target_slot, "NONE"

    # ── 2. Open-door navigation ────────────────────────────────────────────
    open_doors = [d for d in state.doors if d.status == "OPEN" and not d.is_curse]

    if not open_doors:
        # ── 3. No open doors — navigate to a pressure-plate button ──────────
        if state.buttons:
            nearest_btn = min(
                state.buttons,
                key=lambda b: (b.x - state.player_x) ** 2 + (b.y - state.player_y) ** 2,
            )
            if (
                state.player_grid is not None
                and nearest_btn.grid is not None
                and state.grid_w > 0
                and state.grid_size > 0
            ):
                path = astar(
                    state.player_grid, nearest_btn.grid,
                    state.grid_w, state.grid_size, nav_blocked,
                )
                if path and len(path) > 1:
                    return move_from_path_step(path[0], path[1], state.grid_w), nav_target_slot, "NONE"
                if path and len(path) == 1:
                    return "MOVE:STAY", nav_target_slot, "NONE"
                # Fire is blocking path to button — shoot toward it.
                shoot = find_fire_shoot_dir(
                    state.player_grid, nearest_btn.grid, state.grid_w, state.hazards
                )
                if shoot:
                    return "MOVE:STAY", nav_target_slot, shoot
            # Positional fallback (no grid data available).
            dx = nearest_btn.x - state.player_x
            dy = nearest_btn.y - state.player_y
            if abs(dx) < 4 and abs(dy) < 4:
                return "MOVE:STAY", nav_target_slot, "NONE"
            if abs(dx) > abs(dy):
                return ("MOVE:RIGHT" if dx > 0 else "MOVE:LEFT"), nav_target_slot, "NONE"
            return ("MOVE:DOWN" if dy > 0 else "MOVE:UP"), nav_target_slot, "NONE"

        return "MOVE:STAY", nav_target_slot, "NONE"

    if avoid_slot is not None:
        filtered = [d for d in open_doors if d.slot != avoid_slot]
        if filtered:
            open_doors = filtered

    target = None
    if nav_target_slot is not None:
        target = next((d for d in open_doors if d.slot == nav_target_slot), None)

    if target is None:
        target = min(
            open_doors,
            key=lambda d: (d.x - state.player_x) ** 2 + (d.y - state.player_y) ** 2,
        )
        nav_target_slot = target.slot

    dx = target.x - state.player_x
    dy = target.y - state.player_y

    if (
        state.player_grid is not None
        and target.grid is not None
        and state.grid_w > 0
        and state.grid_size > 0
    ):
        path = astar(state.player_grid, target.grid, state.grid_w, state.grid_size, nav_blocked)
        if path and len(path) > 1:
            return move_from_path_step(path[0], path[1], state.grid_w), nav_target_slot, "NONE"
        if path and len(path) == 1:
            return "MOVE:STAY", nav_target_slot, "NONE"
        # Fire is blocking path to door — shoot toward it to clear the way.
        shoot = find_fire_shoot_dir(
            state.player_grid, target.grid, state.grid_w, state.hazards
        )
        if shoot:
            return "MOVE:STAY", nav_target_slot, shoot

    # Positional fallback when no grid info is available.
    if abs(dx) < 4 and abs(dy) < 4:
        return "MOVE:STAY", nav_target_slot, "NONE"
    if abs(dx) > abs(dy):
        return ("MOVE:RIGHT" if dx > 0 else "MOVE:LEFT"), nav_target_slot, "NONE"
    return ("MOVE:DOWN" if dy > 0 else "MOVE:UP"), nav_target_slot, "NONE"


def choose_door_target_slot(state, avoid_slot, door_visit_counts, epsilon):
    open_doors = [d for d in state.doors if d.status == "OPEN" and not d.is_curse]
    if not open_doors:
        return None

    if avoid_slot is not None:
        filtered = [d for d in open_doors if d.slot != avoid_slot]
        if filtered:
            open_doors = filtered

    # Exploration: with epsilon probability, pick a random door to explore new areas early
    if random.random() < epsilon:
        return random.choice(open_doors).slot

    # Exploitation: greedy nearest + least-visited heuristic
    def score(door):
        used = door_visit_counts[(state.room_key, door.slot)]
        dist2 = (door.x - state.player_x) ** 2 + (door.y - state.player_y) ** 2
        return (used, dist2)

    return min(open_doors, key=score).slot


def run_server() -> None:
    cfg = Config()
    ckpt_path = cfg.checkpoint_dir / cfg.checkpoint_name
    run_counter_path = cfg.checkpoint_dir / "run_counter.json"
    run_count = load_and_increment_run_count(run_counter_path)

    log_path = cfg.checkpoint_dir / "training_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_is_new = not log_path.exists()
    log_file = log_path.open("a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if log_is_new:
        log_writer.writerow([
            "run", "episode", "reward", "avg10_reward",
            "steps", "rooms", "duration_s",
            "train_steps", "avg_loss", "epsilon", "buf_size",
        ])
        log_file.flush()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((cfg.host, cfg.port))
    server.listen(1)

    print("--- ISAAC DQN TRAIN SERVER ---")
    print(f"Run #{run_count}")
    print(f"Waiting for Lua bridge on {cfg.host}:{cfg.port} ...")

    conn, addr = server.accept()
    print(f"Connected: {addr}")

    agent = DQNAgent(cfg)
    if agent.load(ckpt_path):
        print(f"Loaded checkpoint: {ckpt_path}")

    last_autosave = time.time()
    last_status_ts = 0.0
    prev_state_vec = None
    prev_action = None
    prev_hp = None
    prev_room_key = None
    prev_in_combat = False
    room_combat_seen = defaultdict(bool)
    room_history = deque(maxlen=3)
    door_visit_counts = defaultdict(int)
    nav_target_slot = None
    entry_slot = None
    last_exit_slot = None
    avoid_slot = None
    room_ticks = 0
    focused_item_key = None
    focused_item_ticks = 0
    unreachable_item_targets = {}
    episode_idx = 1
    episode_reward = 0.0
    episode_steps = 0
    episode_rooms_seen = set()
    episode_start_ts = time.time()
    episode_rewards: deque[float] = deque(maxlen=10)  # rolling window for trend
    recent_losses: deque[float] = deque(maxlen=200)   # rolling avg training loss
    prev_nearest_enemy_dist2 = None
    prev_nearest_enemy_speed = None
    button_stuck_ticks = 0
    prev_player_x = None
    prev_player_y = None
    combat_recent_grids: deque[int] = deque(maxlen=COMBAT_STUCK_WINDOW)
    combat_stuck_ticks = 0
    clear_recent_grids: deque[int] = deque(maxlen=CLEAR_STUCK_WINDOW)
    clear_stuck_ticks = 0

    try:
        while True:
            raw = conn.recv(8192).decode("utf-8", errors="ignore").strip()
            if not raw:
                continue

            latest_line = raw.split("\n")[-1]
            state = parse_packet(latest_line)
            if state is None:
                continue

            room_changed = prev_room_key is not None and state.room_key != prev_room_key
            if room_changed:
                if prev_room_key is not None and last_exit_slot is not None:
                    door_visit_counts[(prev_room_key, last_exit_slot)] += 1
                nav_target_slot = None
                room_ticks = 0
                button_stuck_ticks = 0
                combat_recent_grids.clear()
                combat_stuck_ticks = 0
                clear_recent_grids.clear()
                clear_stuck_ticks = 0
                focused_item_key = None
                focused_item_ticks = 0
                prev_nearest_enemy_dist2 = None
                prev_nearest_enemy_speed = None
                # Identify entry door in the new room and avoid immediate reversal.
                if state.doors:
                    entry_door = min(
                        state.doors,
                        key=lambda d: (d.x - state.player_x) ** 2 + (d.y - state.player_y) ** 2,
                    )
                    entry_slot = entry_door.slot
                else:
                    entry_slot = None
                avoid_slot = entry_slot
            else:
                room_ticks += 1

            in_combat = len(state.enemies) > 0
            state_vec = featurize_state(state)
            room_combat_seen[state.room_key] = room_combat_seen[state.room_key] or in_combat

            if in_combat and state.player_grid is not None:
                combat_recent_grids.append(state.player_grid)
                if (
                    not room_changed
                    and len(combat_recent_grids) >= COMBAT_STUCK_WINDOW
                    and len(set(combat_recent_grids)) <= COMBAT_STUCK_MAX_UNIQUE_GRIDS
                ):
                    combat_stuck_ticks += 1
                else:
                    combat_stuck_ticks = 0
            else:
                combat_recent_grids.clear()
                combat_stuck_ticks = 0

            if (not in_combat) and state.player_grid is not None:
                clear_recent_grids.append(state.player_grid)
                if (
                    not room_changed
                    and len(clear_recent_grids) >= CLEAR_STUCK_WINDOW
                    and len(set(clear_recent_grids)) <= CLEAR_STUCK_MAX_UNIQUE_GRIDS
                ):
                    clear_stuck_ticks += 1
                else:
                    clear_stuck_ticks = 0
            else:
                clear_recent_grids.clear()
                clear_stuck_ticks = 0

            nearest_enemy_dist2 = None
            nearest_enemy_speed = None
            if state.enemies:
                nearest_enemy = min(
                    state.enemies,
                    key=lambda e: (e.x - state.player_x) ** 2 + (e.y - state.player_y) ** 2,
                )
                ndx = nearest_enemy.x - state.player_x
                ndy = nearest_enemy.y - state.player_y
                nearest_enemy_dist2 = ndx * ndx + ndy * ndy
                nearest_enemy_speed = abs(nearest_enemy.vx) + abs(nearest_enemy.vy)

            reward = 0.0
            if prev_hp is not None and state.hp < prev_hp:
                reward -= 100.0
            death_event = prev_hp is not None and prev_hp > 0 and state.hp <= 0
            if death_event:
                reward -= 200.0
            if state.player_grid is not None and state.player_grid in state.hazards:
                # Dense penalty to teach hazard avoidance even before HP drops.
                reward -= HAZARD_CONTACT_PENALTY
            if in_combat and nearest_enemy_dist2 is not None:
                # Penalize body-check distance so the policy learns spacing.
                if nearest_enemy_dist2 < ENEMY_COLLISION_DIST * ENEMY_COLLISION_DIST:
                    reward -= ENEMY_COLLISION_PENALTY
                elif nearest_enemy_dist2 < ENEMY_NEAR_DIST * ENEMY_NEAR_DIST:
                    reward -= ENEMY_NEAR_PENALTY
            if prev_in_combat and not in_combat:
                reward += 75.0
            if prev_room_key is not None and state.room_key != prev_room_key:
                if room_combat_seen.get(prev_room_key, False):
                    reward += 12.0
                if len(room_history) >= 2 and state.room_key == room_history[-2]:
                    reward -= 14.0
            if in_combat and (prev_hp is None or state.hp >= prev_hp):
                reward += 1.0
            if not in_combat:
                reward -= 0.02

            # Generic behavior shaping without hardcoding enemy IDs:
            # chase mostly-stationary enemies, kite fast rushers at close-mid range.
            if in_combat and prev_nearest_enemy_dist2 is not None and nearest_enemy_dist2 is not None:
                prev_speed = prev_nearest_enemy_speed or 0.0
                if prev_speed < 0.35:
                    if nearest_enemy_dist2 < prev_nearest_enemy_dist2:
                        reward += 0.6
                    else:
                        reward -= 0.2
                elif prev_speed > 0.9 and nearest_enemy_dist2 < 170 * 170:
                    if nearest_enemy_dist2 > prev_nearest_enemy_dist2:
                        reward += 0.6
                    else:
                        reward -= 0.25

            episode_reward += reward
            episode_steps += 1
            episode_rooms_seen.add(state.room_key)

            if prev_state_vec is not None and prev_action is not None:
                agent.push_transition(prev_state_vec, prev_action, reward, state_vec, done=death_event)
                loss_val = agent.optimize()
                if loss_val is not None:
                    recent_losses.append(loss_val)

            if death_event:
                # On death, end the episode cleanly and persist progress immediately.
                conn.sendall("RESTART\n".encode("utf-8"))
                now = time.time()
                agent.save(ckpt_path)
                last_autosave = now
                elapsed = now - episode_start_ts
                episode_rewards.append(episode_reward)
                avg_r = sum(episode_rewards) / len(episode_rewards)
                trend = ""
                if len(episode_rewards) >= 3:
                    trend = " (+IMPROVING)" if episode_rewards[-1] > episode_rewards[-2] > episode_rewards[-3] else ""
                avg_loss_val = sum(recent_losses) / len(recent_losses) if recent_losses else None
                avg_loss_str = f"{avg_loss_val:.4f}" if avg_loss_val is not None else "n/a"
                print(
                    f"\n[EPISODE {episode_idx}] DEATH"
                    f" | reward={episode_reward:.1f}  avg10={avg_r:.1f}{trend}"
                    f" | steps={episode_steps}  rooms={len(episode_rooms_seen)}"
                    f" | duration={elapsed:.1f}s  train_steps={agent.train_steps}"
                    f" | avg_loss={avg_loss_str}  eps={agent.epsilon:.4f}"
                    f" | buf={len(agent.replay)}/{agent.replay.capacity}"
                )
                log_writer.writerow([
                    run_count, episode_idx, f"{episode_reward:.2f}", f"{avg_r:.2f}",
                    episode_steps, len(episode_rooms_seen), f"{elapsed:.1f}",
                    agent.train_steps,
                    f"{avg_loss_val:.6f}" if avg_loss_val is not None else "",
                    f"{agent.epsilon:.6f}", len(agent.replay),
                ])
                log_file.flush()

                # Reset per-episode and navigation context; keep learned weights and replay buffer.
                prev_state_vec = None
                prev_action = None
                prev_hp = None
                prev_room_key = None
                prev_in_combat = False
                room_combat_seen.clear()
                room_history.clear()
                nav_target_slot = None
                last_exit_slot = None
                avoid_slot = None
                room_ticks = 0
                focused_item_key = None
                focused_item_ticks = 0
                unreachable_item_targets.clear()
                prev_nearest_enemy_dist2 = None
                prev_nearest_enemy_speed = None
                button_stuck_ticks = 0
                prev_player_x = None
                prev_player_y = None
                combat_recent_grids.clear()
                combat_stuck_ticks = 0
                clear_recent_grids.clear()
                clear_stuck_ticks = 0

                episode_idx += 1
                episode_reward = 0.0
                episode_steps = 0
                episode_rooms_seen.clear()
                episode_start_ts = now
                continue

            now = time.time()
            if now - episode_start_ts >= MAX_EPISODE_SECONDS:
                conn.sendall("RESTART\n".encode("utf-8"))
                agent.save(ckpt_path)
                last_autosave = now
                elapsed = now - episode_start_ts
                episode_rewards.append(episode_reward)
                avg_r = sum(episode_rewards) / len(episode_rewards)
                avg_loss_val = sum(recent_losses) / len(recent_losses) if recent_losses else None
                avg_loss_str = f"{avg_loss_val:.4f}" if avg_loss_val is not None else "n/a"
                print(
                    f"\n[EPISODE {episode_idx}] TIMEOUT-RESTART"
                    f" | reward={episode_reward:.1f}  avg10={avg_r:.1f}"
                    f" | steps={episode_steps}  rooms={len(episode_rooms_seen)}"
                    f" | duration={elapsed:.1f}s  train_steps={agent.train_steps}"
                    f" | avg_loss={avg_loss_str}  eps={agent.epsilon:.4f}"
                    f" | buf={len(agent.replay)}/{agent.replay.capacity}"
                )
                log_writer.writerow([
                    run_count, episode_idx, f"{episode_reward:.2f}", f"{avg_r:.2f}",
                    episode_steps, len(episode_rooms_seen), f"{elapsed:.1f}",
                    agent.train_steps,
                    f"{avg_loss_val:.6f}" if avg_loss_val is not None else "",
                    f"{agent.epsilon:.6f}", len(agent.replay),
                ])
                log_file.flush()

                prev_state_vec = None
                prev_action = None
                prev_hp = None
                prev_room_key = None
                prev_in_combat = False
                room_combat_seen.clear()
                room_history.clear()
                nav_target_slot = None
                last_exit_slot = None
                avoid_slot = None
                room_ticks = 0
                focused_item_key = None
                focused_item_ticks = 0
                unreachable_item_targets.clear()
                prev_nearest_enemy_dist2 = None
                prev_nearest_enemy_speed = None
                button_stuck_ticks = 0
                prev_player_x = None
                prev_player_y = None
                combat_recent_grids.clear()
                combat_stuck_ticks = 0
                clear_recent_grids.clear()
                clear_stuck_ticks = 0

                episode_idx += 1
                episode_reward = 0.0
                episode_steps = 0
                episode_rooms_seen.clear()
                episode_start_ts = now
                continue

            if in_combat:
                button_stuck_ticks = 0
                action_idx = agent.select_action(state_vec)
                move_cmd, shoot_dir = decode_combat_action(action_idx)
                nearest = None
                if state.enemies:
                    nearest = min(
                        state.enemies,
                        key=lambda e: (e.x - state.player_x) ** 2 + (e.y - state.player_y) ** 2,
                    )

                # Combat assist for large rooms:
                # if enemies are far, prioritize obstacle-aware chase so we don't camp one half.
                # Also handles poop-blocked enemies: shoot_override is set when poop needs clearing.
                if nearest is not None:
                    dist2 = (nearest.x - state.player_x) ** 2 + (nearest.y - state.player_y) ** 2
                    chase_move, shoot_override = choose_combat_chase_move(state, nearest)

                    # If direct cardinal line-of-fire is blocked by room geometry,
                    # reposition around the obstacle instead of shooting into it.
                    if (
                        state.player_grid is not None
                        and nearest.grid is not None
                        and state.grid_w > 0
                        and is_cardinal_los_blocked(
                            state.player_grid,
                            nearest.grid,
                            state.grid_w,
                            state.blocked | state.hazards,
                        )
                    ):
                        move_cmd = chase_move

                    if dist2 > 260 ** 2:
                        move_cmd = chase_move
                    elif move_cmd == "MOVE:STAY" and dist2 > 170 ** 2:
                        move_cmd = chase_move
                    if shoot_override is not None:
                        shoot_dir = shoot_override

                    # If we keep circling in a tiny area during combat, break the loop.
                    if combat_stuck_ticks >= COMBAT_STUCK_REPATH_TICKS:
                        move_cmd = chase_move
                        if move_cmd == "MOVE:STAY":
                            move_cmd = random_valid_move(state, allow_stay=False)

                # Guardrail: don't run into walls/hazards when kiting.
                move_cmd = sanitize_move_cmd(state, move_cmd, fallback_cmd=chase_move if nearest is not None else "MOVE:STAY")
                final_cmd = f"{move_cmd};SHOOT:{shoot_dir}"
                prev_state_vec = state_vec
                prev_action = action_idx
            else:
                if nav_target_slot is None:
                    nav_target_slot = choose_door_target_slot(state, avoid_slot, door_visit_counts, agent.epsilon)
                move_cmd, nav_target_slot, shoot_cmd = choose_clear_room_move(
                    state,
                    nav_target_slot,
                    avoid_slot,
                    unreachable_item_targets,
                )

                # If we are parked on an item too long, temporarily skip it to avoid deadlocks.
                if state.items:
                    nearest_item = min(
                        state.items,
                        key=lambda it: (it.x - state.player_x) ** 2 + (it.y - state.player_y) ** 2,
                    )
                    d2 = (nearest_item.x - state.player_x) ** 2 + (nearest_item.y - state.player_y) ** 2
                    item_key = (state.room_key, nearest_item.grid, nearest_item.variant)
                    if d2 < 42 * 42 and item_key == focused_item_key:
                        focused_item_ticks += 1
                    elif d2 < 42 * 42:
                        focused_item_key = item_key
                        focused_item_ticks = 0
                    else:
                        focused_item_key = None
                        focused_item_ticks = 0

                    if focused_item_ticks > 25 and focused_item_key is not None:
                        unreachable_item_targets[focused_item_key] = time.time() + 30.0
                        focused_item_key = None
                        focused_item_ticks = 0

                # Safety guard for non-combat navigation as well.
                door_target_grid = None
                target_door = None
                if nav_target_slot is not None:
                    target_door = next((d for d in state.doors if d.slot == nav_target_slot and d.status == "OPEN"), None)
                    if target_door is not None:
                        door_target_grid = target_door.grid
                # When we're already close to the target door, nudge along the
                # doorway axis to reduce up/down jitter at the threshold.
                if target_door is not None and move_cmd == "MOVE:STAY":
                    near_door = (
                        abs(target_door.x - state.player_x) < 56
                        and abs(target_door.y - state.player_y) < 56
                    )
                    if near_door:
                        move_cmd = door_nudge_move(state.player_x, state.player_y, target_door)
                move_cmd = sanitize_move_cmd(
                    state,
                    move_cmd,
                    fallback_cmd="MOVE:STAY",
                    allowed_target_idx=door_target_grid,
                )

                # Deadlock watchdog for unsolvable button rooms (e.g., button enclosed
                # by rocks/TNT puzzles). Restart only after prolonged no-progress.
                has_open_non_curse = any(d.status == "OPEN" and not d.is_curse for d in state.doors)
                has_buttons = bool(state.buttons)
                button_reachable = False
                if (
                    has_buttons
                    and state.player_grid is not None
                    and state.grid_w > 0
                    and state.grid_size > 0
                ):
                    nav_blocked = state.blocked | state.hazards
                    for b in state.buttons:
                        if b.grid is None:
                            continue
                        path = astar(state.player_grid, b.grid, state.grid_w, state.grid_size, nav_blocked)
                        if path is not None:
                            button_reachable = True
                            break

                no_progress = False
                if prev_player_x is not None and prev_player_y is not None:
                    no_progress = (
                        abs(state.player_x - prev_player_x) <= STUCK_POS_EPS
                        and abs(state.player_y - prev_player_y) <= STUCK_POS_EPS
                    )

                if (
                    not room_changed
                    and not in_combat
                    and has_buttons
                    and not has_open_non_curse
                    and not button_reachable
                    and no_progress
                ):
                    button_stuck_ticks += 1
                else:
                    button_stuck_ticks = 0

                clear_deadlock = (
                    not room_changed
                    and not in_combat
                    and clear_stuck_ticks >= CLEAR_STUCK_RESTART_TICKS
                )

                if button_stuck_ticks >= BUTTON_ROOM_STUCK_RESTART_TICKS or clear_deadlock:
                    conn.sendall("RESTART\n".encode("utf-8"))
                    now = time.time()
                    agent.save(ckpt_path)
                    last_autosave = now
                    elapsed = now - episode_start_ts
                    episode_rewards.append(episode_reward)
                    avg_r = sum(episode_rewards) / len(episode_rewards)
                    avg_loss_val = sum(recent_losses) / len(recent_losses) if recent_losses else None
                    avg_loss_str = f"{avg_loss_val:.4f}" if avg_loss_val is not None else "n/a"
                    restart_reason = "BUTTON-STUCK" if button_stuck_ticks >= BUTTON_ROOM_STUCK_RESTART_TICKS else "CLEAR-STUCK"
                    print(
                        f"\n[EPISODE {episode_idx}] STUCK-RESTART({restart_reason})"
                        f" | reward={episode_reward:.1f}  avg10={avg_r:.1f}"
                        f" | steps={episode_steps}  rooms={len(episode_rooms_seen)}"
                        f" | duration={elapsed:.1f}s  train_steps={agent.train_steps}"
                        f" | avg_loss={avg_loss_str}  eps={agent.epsilon:.4f}"
                        f" | buf={len(agent.replay)}/{agent.replay.capacity}"
                    )
                    log_writer.writerow([
                        run_count, episode_idx, f"{episode_reward:.2f}", f"{avg_r:.2f}",
                        episode_steps, len(episode_rooms_seen), f"{elapsed:.1f}",
                        agent.train_steps,
                        f"{avg_loss_val:.6f}" if avg_loss_val is not None else "",
                        f"{agent.epsilon:.6f}", len(agent.replay),
                    ])
                    log_file.flush()

                    prev_state_vec = None
                    prev_action = None
                    prev_hp = None
                    prev_room_key = None
                    prev_in_combat = False
                    room_combat_seen.clear()
                    room_history.clear()
                    nav_target_slot = None
                    last_exit_slot = None
                    avoid_slot = None
                    room_ticks = 0
                    focused_item_key = None
                    focused_item_ticks = 0
                    unreachable_item_targets.clear()
                    prev_nearest_enemy_dist2 = None
                    prev_nearest_enemy_speed = None
                    button_stuck_ticks = 0
                    prev_player_x = None
                    prev_player_y = None
                    clear_recent_grids.clear()
                    clear_stuck_ticks = 0

                    episode_idx += 1
                    episode_reward = 0.0
                    episode_steps = 0
                    episode_rooms_seen.clear()
                    episode_start_ts = now
                    continue

                final_cmd = f"{move_cmd};SHOOT:{shoot_cmd}"
                prev_state_vec = None
                prev_action = None
                if nav_target_slot is not None and move_cmd.startswith("MOVE:"):
                    last_exit_slot = nav_target_slot

            conn.sendall((final_cmd + "\n").encode("utf-8"))

            now = time.time()
            if now - last_status_ts >= 1.0:
                status = "COMBAT" if in_combat else "CLEAR"
                avg_loss_str = f"{sum(recent_losses)/len(recent_losses):.4f}" if recent_losses else "n/a"
                avg_r_str = f"{sum(episode_rewards)/len(episode_rewards):.1f}" if episode_rewards else "n/a"
                print(
                    f"HP:{state.hp:<3} Room:{state.room_key:<12} En:{len(state.enemies):<2} "
                    f"Proj:{len(state.projectiles):<2} {status:<6} "
                    f"eps:{agent.epsilon:.3f}  loss:{avg_loss_str}  avg_r:{avg_r_str}  "
                    f"steps:{agent.train_steps}  buf:{len(agent.replay)}  Cmd:{final_cmd}"
                )
                last_status_ts = now
            if now - last_autosave >= cfg.autosave_interval_sec:
                agent.save(ckpt_path)
                last_autosave = now

            prev_hp = state.hp
            prev_room_key = state.room_key
            prev_in_combat = in_combat
            prev_nearest_enemy_dist2 = nearest_enemy_dist2
            prev_nearest_enemy_speed = nearest_enemy_speed
            prev_player_x = state.player_x
            prev_player_y = state.player_y
            if room_changed:
                room_history.append(state.room_key)
            elif not room_history or room_history[-1] != state.room_key:
                room_history.append(state.room_key)

            if not cfg.train_fast and cfg.loop_sleep_sec > 0:
                time.sleep(cfg.loop_sleep_sec)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint...")
    finally:
        agent.save(ckpt_path)
        log_file.close()
        conn.close()
        server.close()
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    run_server()

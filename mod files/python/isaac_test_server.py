import socket
import threading
import sys
import os
import random
import json
import time
from collections import defaultdict

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

# Learning Hyperparameters
alpha = 0.1    # Learning Rate (how fast it learns)
gamma = 0.9    # Discount Factor (how much it values future rewards)
epsilon = 0.1  # Exploration Rate (10% chance to try something random)


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


def move_toward_target(p_pos, target):
    """Converts a target point into a movement command."""
    if target is None:
        return "MOVE:STAY"

    dx = target["x"] - p_pos[0]
    dy = target["y"] - p_pos[1]

    # Stop if we are close enough to the door (prevents vibrating)
    if abs(dx) < 15 and abs(dy) < 15:
        return "MOVE:STAY"

    # Walk in the direction of the biggest gap
    if abs(dx) > abs(dy):
        return "MOVE:RIGHT" if dx > 0 else "MOVE:LEFT"
    return "MOVE:DOWN" if dy > 0 else "MOVE:UP"


# Load persisted Q-values before starting the real-time loop.
load_brain()

# --- 3. THE BRAIN (DATA PROCESSING) ---
def listen_to_isaac(conn):
    # Memory variables for the Markov Process
    # These track transition context needed for reward shaping and anti-loop navigation.
    last_state = "BOOT"
    last_action_idx = COMBAT_ACTIONS.index("MOVE:STAY")
    last_hp = None
    last_room_key = None
    last_exit_slot = None

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
            p_pos = [float(x) for x in parts[0][2:].split(",")]
            hp = int(parts[1][2:])
            room_key = "UNKNOWN"
            
            # SENSING: Enemies & Doors
            enemies = []
            doors = []
            for p in parts:
                if p.startswith("E:"):
                    e_parts = p.split(":")
                    enemies.append({"id": e_parts[1], "x": float(e_parts[2].split(",")[0]), "y": float(e_parts[2].split(",")[1])})
                if p.startswith("D:"):
                    # Split the door string so we can read status
                    d_info = p.split(":") 
                    # Capture status AND the X,Y coordinates
                    coords = d_info[3].split(",")
                    doors.append({
                        "slot": int(d_info[1]),
                        "status": d_info[2], 
                        "x": float(coords[0]), 
                        "y": float(coords[1])
                    })
                if p.startswith("R:"):
                    r_info = p.split(":")
                    if len(r_info) >= 3:
                        room_key = f"{r_info[1]}:{r_info[2]}"

            # --- STATE & AIM MATH ---
            # On room transitions, avoid the opposite of the previous exit door when possible.
            room_changed = last_room_key is not None and room_key != last_room_key
            avoid_slot = None
            if room_changed and last_exit_slot is not None:
                avoid_slot = OPPOSITE_DOOR.get(last_exit_slot)

            target_dir = "NONE"
            if enemies:
                closest = min(enemies, key=lambda e: (e['x']-p_pos[0])**2 + (e['y']-p_pos[1])**2)
                dx, dy = closest['x'] - p_pos[0], closest['y'] - p_pos[1]
                
                rel_x = "RIGHT" if dx > 0 else "LEFT"
                rel_y = "DOWN" if dy > 0 else "UP"
                name = ENEMY_NAMES.get(closest['id'], f"ID_{closest['id']}")
                
                current_state = f"COMBAT_{name}_{rel_x}_{rel_y}"
                target_dir = rel_x if abs(dx) > abs(dy) else rel_y

                # --- DECIDE NEXT ACTION (epsilon-greedy) ---
                if random.random() < epsilon:
                    action_idx = random.randint(0, len(COMBAT_ACTIONS) - 1)
                else:
                    action_idx = q_table[current_state].index(max(q_table[current_state]))

                # --- EXECUTE ACTION ---
                chosen_move = COMBAT_ACTIONS[action_idx]
                if chosen_move == "SHOOT":
                    final_cmd = f"SHOOT:{target_dir}" if target_dir != "NONE" else "SHOOT:NONE"
                else:
                    final_cmd = chosen_move

            else:
                # --- ROOM IS CLEAR: navigate to a useful door ---
                # Encode room context in state so navigation can also be learned over time.
                open_doors = [d for d in doors if d["status"] == "OPEN"]
                current_state = f"CLEAR_{room_key}_OD{len(open_doors)}"

                target = choose_nav_target(p_pos, doors, avoid_slot)
                heuristic_cmd = move_toward_target(p_pos, target)

                # Navigation still uses the Q-table, with a heuristic fallback while untrained.
                if random.random() < epsilon:
                    action_idx = random.randint(0, len(MOVE_ACTIONS) - 1)
                    final_cmd = MOVE_ACTIONS[action_idx]
                else:
                    nav_values = q_table[current_state][:len(MOVE_ACTIONS)]
                    best_nav_idx = nav_values.index(max(nav_values))
                    if max(nav_values) == 0.0:
                        final_cmd = heuristic_cmd
                        action_idx = COMBAT_ACTIONS.index(final_cmd)
                    else:
                        action_idx = best_nav_idx
                        final_cmd = MOVE_ACTIONS[action_idx]

                if target and final_cmd == "MOVE:STAY":
                    # We likely touched this door; remember it so next room can avoid immediate reversal.
                    last_exit_slot = target["slot"]

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

            msg += f"Enemies: {len(enemies):<3} | Open Doors: {open_doors_count:<2} | Reward: {reward:<5} | STATUS: {status_msg} | Cmd: {final_cmd:10}"
            
            sys.stdout.write(msg)
            sys.stdout.flush()

            # Update Memory
            last_state, last_action_idx, last_hp = current_state, action_idx, hp
            last_room_key = room_key
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
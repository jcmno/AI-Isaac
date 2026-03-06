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

# Lists Enemy names (does not include ALL names, only some)
ENEMY_NAMES = {"10.0": "Gaper", "13.0": "Fly", "85.0": "Monstro"}

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


# Get Isaac to move towards an open door 
def get_nav_to_door(p_pos, doors):
    # Only look for doors that are actually OPEN
    open_doors = [d for d in doors if d['status'] == "OPEN"]
    if not open_doors: 
        return "MOVE:STAY"
    
    
    # Take the first open door found
    # **BIG ISSUE HERE** Causes an infinite loop because it can take the same door over and over again. 
    # **NEEDS FIXING**
    target = open_doors[0]
    
    dx = target['x'] - p_pos[0]
    dy = target['y'] - p_pos[1]

    # Stop if we are close enough to the door (prevents vibrating)
    if abs(dx) < 15 and abs(dy) < 15: 
        return "MOVE:STAY"

    # Walk in the direction of the biggest gap
    if abs(dx) > abs(dy):
        return "MOVE:RIGHT" if dx > 0 else "MOVE:LEFT"
    else:
        return "MOVE:DOWN" if dy > 0 else "MOVE:UP"

# --- 3. THE BRAIN (DATA PROCESSING) ---
def listen_to_isaac(conn):

    # Memory variables for the Markov Process
    last_state = "CLEAR"
    last_action_idx = 0
    last_hp = 0

    while True:
        try:
            raw_data = conn.recv(4096).decode('utf-8').strip()
            if not raw_data: continue

            # Get the most recent packet
            latest_line = raw_data.split('\n')[-1]
            parts = latest_line.split("|")
            
            # P:X,Y | H:HP
            p_pos = [float(x) for x in parts[0][2:].split(",")]
            hp = int(parts[1][2:])
            
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
                        "status": d_info[2], 
                        "x": float(coords[0]), 
                        "y": float(coords[1])
                    })

            # --- STATE & AIM MATH ---
            target_dir = "NONE"
            if enemies:
                closest = min(enemies, key=lambda e: (e['x']-p_pos[0])**2 + (e['y']-p_pos[1])**2)
                dx, dy = closest['x'] - p_pos[0], closest['y'] - p_pos[1]
                
                rel_x = "RIGHT" if dx > 0 else "LEFT"
                rel_y = "DOWN" if dy > 0 else "UP"
                name = ENEMY_NAMES.get(closest['id'], f"ID_{closest['id']}")
                
                current_state = f"{name}_{rel_x}_{rel_y}"
                target_dir = rel_x if abs(dx) > abs(dy) else rel_y
            

                # --- REWARD CALCULATION ---
                reward = 0
                if hp < last_hp: reward = -100
                elif current_state == "CLEAR" and last_state != "CLEAR": reward = 100
                elif current_state != "CLEAR": reward = 1
                
                # --- Q-LEARNING UPDATE ---
                old_q = q_table[last_state][last_action_idx]
                max_future_q = max(q_table[current_state])
                q_table[last_state][last_action_idx] = old_q + alpha * (reward + gamma * max_future_q - old_q)

                # --- DECIDE NEXT ACTION ---
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
                # [NEW NAVIGATION LOGIC]
                # --- ROOM IS CLEAR: GO TO DOOR ---
                current_state = "CLEAR"
                
                
                # Use the math function to find the way out
                final_cmd = get_nav_to_door(p_pos, doors) 
                
                reward = 0 
                action_idx = 4 # Match the index for MOVE:STAY
                status_msg = "NAVIGATING"

            conn.sendall(f"{final_cmd}\n".encode('utf-8'))

            # Convert raw coordinates to "Grid" units (approx 40px per tile)
            grid_x = int(p_pos[0] / 40)
            grid_y = int(p_pos[1] / 40)
            open_doors_count = len([d for d in doors if d['status'] == "OPEN"])
            
            # Line 1: Main Stats
            msg = f"\033[H[LIVE DATA] HP: {hp} | Grid: ({grid_x},{grid_y}) | State: {current_state:15}\n"
            
            # Line 2: Room Info
            if not enemies:
                status_msg = "ROOM CLEAR" if open_doors_count > 0 else "WAITING..."
            else:
                status_msg = "IN COMBAT "

            msg += f"Enemies: {len(enemies):<3} | Open Doors: {open_doors_count:<2} | Reward: {reward:<5} | STATUS: {status_msg}        "
            
            sys.stdout.write(msg)
            sys.stdout.flush()

            # Update Memory
            last_state, last_action_idx, last_hp = current_state, action_idx, hp
            time.sleep(0.1) # Keeps movement from being "sticky"
                
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
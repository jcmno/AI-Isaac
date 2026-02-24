import socket
import threading
import sys
import os

# 1. Setup Server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('127.0.0.1', 5005))
server.listen(1)
os.system('cls' if os.name == 'nt' else 'clear') # Clear screen for clean start
print("--- ISAAC AI COMMANDER ---")
print("Waiting for Isaac...")

conn, addr = server.accept()
print(f"Link established with {addr}!")

# 2. THE BRAIN: Vision & Tracking
def listen_to_isaac():
    while True:
        try:
            raw_data = conn.recv(4096).decode('utf-8').strip()
            if not raw_data: continue

            latest_line = raw_data.split('\n')[-1]
            parts = latest_line.split("|")
            
            p_raw = parts[0].replace("P:", "").split(",")
            if len(p_raw) == 2:
                player_display = f"X: {p_raw[0]}, Y: {p_raw[1]}"
            else:
                player_display = "Scanning..."

            enemies = [p for p in parts if p.startswith("E:")]

            # THE FIX: This moves the cursor to the top of the screen (0,0), 
            # prints the status, then returns to the bottom so you can type.
            msg = f"\033[H[LIVE DATA] Player: ({player_display}) | Enemies: {len(enemies)}!      "
            sys.stdout.write(msg)
            sys.stdout.flush()
                
        except Exception:
            continue

# Start the thread
threading.Thread(target=listen_to_isaac, daemon=True).start()

# 3. THE HANDS: Main command loop
print("\n" * 3) # Move the cursor down so it doesn't overlap the status
try:
    while True:
        # This prompt will now stay at the bottom
        cmd = input(">> ENTER COMMAND (spawn/hurt/speed/tp): ").strip().lower()
        if cmd:
            conn.sendall((cmd + "\n").encode('utf-8'))
            # Clear the 'Sent' line quickly so it doesn't pile up
            sys.stdout.write(f"\033[KLast Action: {cmd}\n") 
except KeyboardInterrupt:
    print("\nClosing...")
finally:
    conn.close()
    server.close()
 
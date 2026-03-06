# The Binding of Isaac - AI Mod
Purpose: Build a working game-to-Python interface and a basic reinforcement-learning loop that turns room state into actions every frame. This is an integration milestone, not a fully trained bot.

## Requirements
- The Binding of Isaac: Repentence (Steam Version)
- Repentogon (Mod Extension)

## Set-up Instructions 
1. Install The Binding of Isaac: Repentence 
2. Install Repentogon 
- Click [here](https://youtu.be/hF4ngfDn364?si=qrt4d8w2WkdSY-hs) for a step-by-step walkthrough
- Add '--luadebug' to your Steam launch options
- Launch the game through Steam 
3. Mod Folders 
- Locate the game files (usually located in the x86 Program Files folder)
- Open the 'mods' folder 
- Copy and paste the **PythonBridge** folder from this repo into the 'mods' folder 

## Running the Mod
1. Run the "Isaac_test_server.py" file on the terminal. The command prompt should say "Waiting for Isaac...".
2. Launch The Binding of Isaac 
3. Start a game
4. Once loaded in the game, "Link established!" appears in Python and live data updates begin.
5. The loop runs continuously: Lua sends room/player/enemy/door data, Python chooses an action, and Lua executes it.
6. Manual commands in Python: "save" to persist the Q-table, "exit" to stop safely.

## Current milestone behavior
- Demonstrates bidirectional socket communication between Isaac (Lua) and Python.
- Demonstrates a per-tick RL decision loop with Q-table updates.
- Includes simple anti-backtracking door navigation to reduce empty-room ping-pong.
- Navigation policy is still early-stage and not fully trained.

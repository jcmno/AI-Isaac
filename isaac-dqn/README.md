# Isaac DQN (PyTorch)

A clean starter project for training a neural-network combat agent for Binding of Isaac.

## What This Project Does

- Runs a Python socket server on `127.0.0.1:5005`
- Parses telemetry from your Lua bridge (`P`, `H`, `R`, `E`, `T`, `G`, `Z`)
- Trains a small DQN model for combat movement decisions
- Sends combined commands like `MOVE:LEFT;SHOOT:RIGHT`
- Saves model checkpoints periodically and on exit

## Project Layout

- `src/config.py`: runtime settings
- `src/protocol.py`: packet parser and typed state objects
- `src/features.py`: state feature extraction
- `src/dqn.py`: neural network definition
- `src/replay_buffer.py`: experience replay buffer
- `src/agent.py`: DQN training logic
- `src/train_server.py`: main socket training loop
- `scripts/run_train.ps1`: helper script to start training

## Setup

```powershell
cd "c:\Users\eulun\Desktop\310 Project\AI-Isaac\isaac-dqn"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

## Run

```powershell
cd "c:\Users\eulun\Desktop\310 Project\AI-Isaac\isaac-dqn"
py -m src.train_server
```

Or:

```powershell
.\scripts\run_train.ps1
```

## Notes

- This starter trains combat movement only (5 move actions).
- Shooting direction is still heuristic (nearest enemy direction) to keep first version stable.
- Model is saved to `checkpoints/isaac_dqn.pt`.

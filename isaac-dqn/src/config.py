from dataclasses import dataclass
from pathlib import Path


MOVE_ACTIONS = ["MOVE:UP", "MOVE:DOWN", "MOVE:LEFT", "MOVE:RIGHT", "MOVE:STAY"]
SHOOT_ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "NONE"]
COMBAT_ACTIONS = [(m, s) for m in MOVE_ACTIONS for s in SHOOT_ACTIONS]


def decode_combat_action(action_idx: int) -> tuple[str, str]:
    action_idx = max(0, min(action_idx, len(COMBAT_ACTIONS) - 1))
    return COMBAT_ACTIONS[action_idx]


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 5005

    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 50_000
    target_sync_steps: int = 1000

    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.990

    autosave_interval_sec: int = 120
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_name: str = "isaac_dqn.pt"

    hidden_size: int = 128
    feature_dim: int = 32
    action_dim: int = len(COMBAT_ACTIONS)

    train_fast: bool = True
    loop_sleep_sec: float = 0.0

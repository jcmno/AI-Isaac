from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import Config
from .dqn import DQN
from .replay_buffer import ReplayBuffer, Transition


class DQNAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cpu")

        self.policy_net = DQN(cfg.feature_dim, cfg.action_dim, cfg.hidden_size).to(self.device)
        self.target_net = DQN(cfg.feature_dim, cfg.action_dim, cfg.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(cfg.replay_capacity)

        self.epsilon = cfg.epsilon_start
        self.train_steps = 0

    def select_action(self, state_vec: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.cfg.action_dim))
        with torch.no_grad():
            s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(s)
            return int(torch.argmax(q_values, dim=1).item())

    def push_transition(
        self,
        state_vec: np.ndarray,
        action: int,
        reward: float,
        next_state_vec: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.push(
            Transition(
                state=state_vec,
                action=action,
                reward=float(reward),
                next_state=next_state_vec,
                done=done,
            )
        )

    def optimize(self) -> float | None:
        if len(self.replay) < self.cfg.batch_size:
            return None

        batch = self.replay.sample(self.cfg.batch_size)

        states = torch.tensor(
            np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(
            np.stack([t.next_state for t in batch]), dtype=torch.float32, device=self.device
        )
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True).values
            q_target = rewards + (1.0 - dones) * self.cfg.gamma * max_next_q

        loss = self.loss_fn(q_sa, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_sync_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)
        return float(loss.item())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy_net.state_dict(),
                "target": self.target_net.state_dict(),
                "epsilon": self.epsilon,
                "train_steps": self.train_steps,
            },
            path,
        )

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            data = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(data["policy"])
            self.target_net.load_state_dict(data["target"])
            self.epsilon = float(data.get("epsilon", self.cfg.epsilon_start))
            self.train_steps = int(data.get("train_steps", 0))
            return True
        except (RuntimeError, KeyError, TypeError, ValueError):
            # Architecture or checkpoint format mismatch; start fresh.
            return False

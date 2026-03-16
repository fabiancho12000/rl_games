"""
Deep Q-Network (DQN) implementation in PyTorch.

This module intentionally avoids high-level RL libraries so every piece of
the algorithm is visible and editable for learning purposes.

Key components:
  - QNetwork     : a small fully-connected network that maps state → Q(s,a)
  - ReplayBuffer : stores (s, a, r, s', done) transitions for experience replay
  - DQNAgent     : the training loop, ε-greedy policy, and target-network sync
"""
import random
from collections import deque
from pathlib import Path
from typing import Self

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ── Neural network ────────────────────────────────────────────────────


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay buffer ────────────────────────────────────────────────────


class ReplayBuffer:
    """Fixed-size FIFO buffer that stores transitions for experience replay."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ── Agent ─────────────────────────────────────────────────────────────


class DQNAgent:
    """
    Deep Q-Network agent implemented from scratch.

    Hyperparameters are intentionally exposed as constructor args so you
    can experiment with them directly.
    """

    def __init__(
        self,
        env_id: str,
        *,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.997,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 1000,
        hidden: int = 128,
        min_buffer_size: int = 5000,
        train_freq: int = 4,
    ) -> None:
        self.env_id = env_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_buffer_size = min_buffer_size
        self.train_freq = train_freq
        self.training_episodes = 0
        self.total_steps = 0

        env = gym.make(env_id)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = int(env.action_space.n)  # type: ignore[attr-defined]
        env.close()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_capacity)

    # ── policy ────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, *, deterministic: bool = False) -> int:
        if not deterministic and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(t)
            return int(q_values.argmax(dim=1).item())

    def predict(
        self, obs: np.ndarray, *, deterministic: bool = True
    ) -> tuple[int, None]:
        return self.select_action(obs, deterministic=deterministic), None

    # ── learning step ─────────────────────────────────────────────────

    def _learn(self) -> float:
        """Sample a mini-batch from the buffer and perform one gradient step.

        Returns the batch loss value.
        """
        if len(self.buffer) < max(self.batch_size, self.min_buffer_size):
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.q_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        return float(loss.item())

    # ── training loop ─────────────────────────────────────────────────

    def train(self, total_episodes: int = 1000, log_interval: int = 10) -> list[float]:
        env = gym.make(self.env_id)
        rewards_history: list[float] = []
        best_reward = float("-inf")
        best_avg = float("-inf")

        for episode in range(1, total_episodes + 1):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            episode_loss = []

            while not done:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.buffer.push(obs, action, float(reward), next_obs, done)

                self.total_steps += 1

                if self.total_steps % self.train_freq == 0:
                    loss = self._learn()
                    if loss > 0:
                        episode_loss.append(loss)

                if self.total_steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

                obs = next_obs
                total_reward += reward

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.training_episodes += 1
            rewards_history.append(total_reward)

            if total_reward > best_reward:
                best_reward = total_reward

            if len(rewards_history) >= log_interval:
                current_avg = float(np.mean(rewards_history[-log_interval:]))
                if current_avg > best_avg:
                    best_avg = current_avg

            if episode % log_interval == 0:
                avg = np.mean(rewards_history[-log_interval:])
                mean_loss = float(np.mean(episode_loss)) if episode_loss else 0.0
                print(
                    f"Episode {episode}/{total_episodes} | "
                    f"Avg Reward: {avg:.2f} | "
                    f"Best Reward: {best_reward:.2f} | "
                    f"Best Avg({log_interval}): {best_avg:.2f} | "
                    f"Epsilon: {self.epsilon:.4f} | "
                    f"Buffer: {len(self.buffer)} | "
                    f"Steps: {self.total_steps} | "
                    f"Loss: {mean_loss:.4f}"
                )

        env.close()

        final_avg_100 = float(np.mean(rewards_history[-100:])) if len(rewards_history) >= 100 else float(np.mean(rewards_history))
        final_avg_500 = float(np.mean(rewards_history[-500:])) if len(rewards_history) >= 500 else float(np.mean(rewards_history))

        print("\nTraining finished")
        print(f"Final Avg(100): {final_avg_100:.2f}")
        print(f"Final Avg(500): {final_avg_500:.2f}")
        print(f"Best Reward: {best_reward:.2f}")
        print(f"Best Avg({log_interval}): {best_avg:.2f}")
        print(f"Total steps: {self.total_steps}")
        print(f"Replay buffer size: {len(self.buffer)}")

        return rewards_history

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_net_state": self.q_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_episodes": self.training_episodes,
            "env_id": self.env_id,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "min_buffer_size": self.min_buffer_size,
            "train_freq": self.train_freq,
            "total_steps": self.total_steps,
        }
        torch.save(data, path)
        print(f"Saved DQN agent to {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        data = torch.load(path, weights_only=False)
        agent = cls(
            data["env_id"],
            lr=data["lr"],
            gamma=data["gamma"],
            epsilon_start=data["epsilon"],
            epsilon_end=data["epsilon_end"],
            epsilon_decay=data["epsilon_decay"],
            batch_size=data["batch_size"],
            target_update_freq=data["target_update_freq"],
            min_buffer_size=data.get("min_buffer_size", 5000),
            train_freq=data.get("train_freq", 4),
        )
        agent.q_net.load_state_dict(data["q_net_state"])
        agent.target_net.load_state_dict(data["target_net_state"])
        agent.optimizer.load_state_dict(data["optimizer_state"])
        agent.training_episodes = data["training_episodes"]
        agent.total_steps = data.get("total_steps", 0)
        return agent

    def info(self) -> str:
        params = sum(p.numel() for p in self.q_net.parameters())
        return (
            f"DQN agent for {self.env_id}\n"
            f"  Episodes trained  : {self.training_episodes}\n"
            f"  Network params    : {params:,}\n"
            f"  Epsilon           : {self.epsilon:.4f}\n"
            f"  LR / Gamma        : {self.lr} / {self.gamma}\n"
            f"  Batch size        : {self.batch_size}\n"
            f"  Min buffer size   : {self.min_buffer_size}\n"
            f"  Train frequency   : every {self.train_freq} steps\n"
            f"  Target update     : every {self.target_update_freq} steps\n"
            f"  Device            : {self.device}"
        )
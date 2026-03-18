import pickle
from collections import defaultdict
from pathlib import Path
from typing import Self

import gymnasium as gym
import numpy as np

# Approximate observation bounds
# Dims 0-5 are continuous; dims 6-7 are binary leg-contact flags.
_OBS_BOUNDS = np.array(
    [
        [-1.5, 1.5],
        [-0.5, 1.5],
        [-5.0, 5.0],
        [-5.0, 5.0],
        [-3.14, 3.14],
        [-5.0, 5.0],
    ],
    dtype=np.float32,
)

_OBS_LOW = _OBS_BOUNDS[:, 0]
_OBS_HIGH = _OBS_BOUNDS[:, 1]

_N_ACTIONS = 4


class QLearningAgent:
    def __init__(
        self,
        env_id: str,
        *,
        n_bins: int = 6,
        lr: float = 0.10,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9998,
    ) -> None:
        self.env_id = env_id
        self.n_bins = n_bins
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.training_episodes = 0

        self._bins = [
            np.linspace(lo, hi, n_bins + 1, dtype=np.float32)[1:-1]
            for lo, hi in _OBS_BOUNDS
        ]
        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(_N_ACTIONS, dtype=np.float32)
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def discretize(self, obs: np.ndarray) -> tuple:
        cont = obs[:6]

        indices = [
            int(np.digitize(
                _OBS_LOW[i] if cont[i] < _OBS_LOW[i]
                else _OBS_HIGH[i] if cont[i] > _OBS_HIGH[i]
                else cont[i],
                self._bins[i]
            ))
            for i in range(6)
        ]

        return (
            indices[0],
            indices[1],
            indices[2],
            indices[3],
            indices[4],
            indices[5],
            int(obs[6]),
            int(obs[7]),
        )

    def select_action(self, state: tuple, *, deterministic: bool = False) -> int:
        if not deterministic and np.random.random() < self.epsilon:
            return int(np.random.randint(_N_ACTIONS))
        return int(np.argmax(self.q_table[state]))

    def predict(
        self, obs: np.ndarray, *, deterministic: bool = True
    ) -> tuple[int, None]:
        state = self.discretize(obs)
        return self.select_action(state, deterministic=deterministic), None

    # ------------------------------------------------------------------
    # core RL
    # ------------------------------------------------------------------

    def _update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> None:
        q_table = self.q_table
        state_q = q_table[state]

        best_next = 0.0 if done else float(np.max(q_table[next_state]))
        td_target = reward + self.gamma * best_next
        state_q[action] += self.lr * (td_target - state_q[action])

    def train(
        self,
        total_episodes: int = 20_000,
        log_interval: int = 100
    ) -> list[float]:
        env = gym.make(self.env_id)
        rewards_history: list[float] = []
        best_reward = float("-inf")
        best_avg = float("-inf")

        discretize = self.discretize
        select_action = self.select_action
        update = self._update
        q_table = self.q_table
        epsilon_end = self.epsilon_end
        epsilon_decay = self.epsilon_decay

        for episode in range(1, total_episodes + 1):
            obs, _ = env.reset()
            state = discretize(obs)
            total_reward = 0.0
            done = False

            while not done:
                action = select_action(state)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                next_state = discretize(next_obs)
                update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            self.epsilon = max(epsilon_end, self.epsilon * epsilon_decay)
            self.training_episodes += 1
            rewards_history.append(total_reward)

            if total_reward > best_reward:
                best_reward = total_reward

            if len(rewards_history) >= log_interval:
                recent = rewards_history[-log_interval:]
                current_avg = float(np.mean(recent))
                if current_avg > best_avg:
                    best_avg = current_avg

            if episode % log_interval == 0:
                avg = float(np.mean(rewards_history[-log_interval:]))
                print(
                    f"Episode {episode}/{total_episodes} | "
                    f"Avg Reward: {avg:.2f} | "
                    f"Best Reward: {best_reward:.2f} | "
                    f"Best Avg({log_interval}): {best_avg:.2f} | "
                    f"Epsilon: {self.epsilon:.4f} | "
                    f"States visited: {len(q_table)}"
                )

        env.close()
        return rewards_history

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "training_episodes": self.training_episodes,
            "env_id": self.env_id,
            "n_bins": self.n_bins,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved Q-Learning agent to {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        agent = cls(
            env_id=data["env_id"],
            n_bins=data["n_bins"],
            lr=data["lr"],
            gamma=data["gamma"],
            epsilon_start=data["epsilon"],
            epsilon_end=data["epsilon_end"],
            epsilon_decay=data["epsilon_decay"],
        )
        agent.q_table = defaultdict(
            lambda: np.zeros(_N_ACTIONS, dtype=np.float32),
            data["q_table"]
        )
        agent.training_episodes = data["training_episodes"]
        return agent

    def info(self) -> str:
        return (
            f"Q-Learning agent for {self.env_id}\n"
            f"  Episodes trained : {self.training_episodes}\n"
            f"  States visited   : {len(self.q_table)}\n"
            f"  Epsilon          : {self.epsilon:.4f}\n"
            f"  LR / Gamma       : {self.lr} / {self.gamma}"
        )
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from reward_func import quadcopter_reward

class QuadcopterEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        # 12D state array
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        # 4D action array
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.max_steps = 500
        self.goal_state = np.zeros(12) 
        self.state = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pos = np.random.uniform(-2, 2, size=3)
        angles = np.random.uniform(-0.1, 0.1, size=3)
        vel = np.random.uniform(-0.05, 0.05, size=3)
        ang_vel = np.random.uniform(-0.05, 0.05, size=3)
        self.state = np.concatenate([pos, angles, vel, ang_vel]).astype(np.float32)
        self.steps = 0
        return self.state, {}
    
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.state = self.state + 0.05 * np.concatenate([action, np.zeros(8)])
        self.steps += 1

        reward = quadcopter_reward(self.state, action, goal_state=self.goal_state)
        out_of_bounds = np.linalg.norm(self.state[:3]) > 10
        terminated = out_of_bounds
        truncated = self.steps >= self.max_steps
        done = terminated or truncated

        info = {"out_of_bounds": out_of_bounds}

        return self.state, reward, done, info


    def render(self):
        print(f"State: {self.state}")

    def close(self):
        pass

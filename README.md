# Quadcopter RL Environment

This repository provides a custom OpenAI Gymnasium environment for simulating and training reinforcement learning (RL) agents to control a quadcopter. It includes a reward function, environment implementation, and a sample notebook for training with the TD3 algorithm using Stable Baselines3.

## Features
- **Custom Environment**: `QuadcopterEnv` simulates a 12-dimensional quadcopter state and 4-dimensional action space.
- **Reward Function**: Flexible quadratic reward function for state and action penalties.
- **TD3 Training Example**: Jupyter notebook (`td3.ipynb`) demonstrates training and evaluation of a TD3 agent.
- **Logging**: Training logs are saved in the `logs/` directory for analysis.

## File Overview
- `quad_copter.py`: Defines the `QuadcopterEnv` class, a Gymnasium-compatible environment for quadcopter control.
- `reward_func.py`: Contains the `quadcopter_reward` function, a quadratic cost-based reward for RL.
- `td3.ipynb`: Jupyter notebook for training and evaluating a TD3 agent on the custom environment.
- `logs/`: Directory for training logs and monitor files.

## Getting Started

### Prerequisites
- Python 3.8+
- [gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/) (for log analysis)

Install dependencies:
```bash
pip install gymnasium stable-baselines3 numpy pandas
```

### Usage
1. **Custom Environment**: Use `QuadcopterEnv` from `quad_copter.py` in your RL experiments.
2. **Reward Function**: Import and use `quadcopter_reward` for custom reward shaping.
3. **Training**: Run the `td3.ipynb` notebook to train and evaluate a TD3 agent.

### Example
```python
from quad_copter import QuadcopterEnv
import numpy as np

env = QuadcopterEnv()
obs, _ = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        break
```

## Notes
- The environment state is a 12D vector: position, angles, velocities, and angular velocities.
- The action is a 4D vector, typically representing motor commands.
- The reward penalizes deviation from the goal state and large actions.

## License
MIT License

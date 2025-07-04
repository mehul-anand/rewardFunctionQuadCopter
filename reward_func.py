import numpy as np

def quadcopter_reward(state, action, Q=None, R=None, goal_state=None):
    """
    quad reward function for quadcopter RL.
    state: 12D array
    action: 4D array
    goal_state: origin
    """
    state = np.asarray(state)
    action = np.asarray(action)
    if goal_state is None:
        goal_state = np.zeros_like(state)
    state_error = state - goal_state

    if Q is None:
        Q = np.eye(12)
    elif Q.ndim == 1:
        Q = np.diag(Q)
    if R is None:
        R = np.eye(4)
    elif R.ndim == 1:
        R = np.diag(R)

    # quad cost
    state_cost = state_error.T @ Q @ state_error
    action_cost = action.T @ R @ action

    # minimise    
    reward = - (state_cost + action_cost)
    return reward

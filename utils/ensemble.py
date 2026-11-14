"""
Temporal Ensemble.
"""

import threading
import numpy as np
from collections import deque


class EnsembleBuffer:
    """
    Temporal ensemble buffer.
    """
    def __init__(self, mode = "new", **kwargs):
        assert mode in ["new", "old", "avg", "act", "hato"], "Ensemble mode {} not supported.".format(mode)
        self.mode = mode
        self.timestep = 0
        self.actions_start_timestep = 0
        self.actions = deque([])
        self.actions_timestep = deque([])
        self.action_shape = None
        self._lock = threading.Lock()
        self.max_past_inference_chunks = kwargs.get("max_past_inference_chunks", None)
        if self.max_past_inference_chunks is not None:
            self.max_past_inference_chunks = int(self.max_past_inference_chunks)
            assert self.max_past_inference_chunks >= 0, "max_past_inference_chunks must be non-negative."
        if mode == "act":
            self.k = kwargs.get("k", 0.01)
        if mode == "hato":
            self.tau = kwargs.get("tau", 0.5)

    def peek_timestep(self):
        """Next control-step index (same as internal timestep before the next get_action). Thread-safe."""
        with self._lock:
            return self.timestep
    
    def add_action(self, action, timestep):
        """
        Add action to the ensemble buffer:

        Parameters:
        - action: horizon x action_dim (...);
        - timestep: action[0]'s timestep.
        """
        action = np.array(action)
        with self._lock:
            self._add_action_unlocked(action, timestep)

    def _add_action_unlocked(self, action, timestep):
        if self.action_shape == None:
            self.action_shape = action.shape[1:]
        else:
            assert self.action_shape == action.shape[1:], "Incompatible action shape."
        idx = timestep - self.actions_start_timestep
        horizon = action.shape[0]
        while idx + horizon - 1 >= len(self.actions):
            self.actions.append([])
            self.actions_timestep.append([])
        for i in range(idx, idx + horizon):
            self.actions[i].append(action[i - idx, ...])
            self.actions_timestep[i].append(timestep)

    def get_action(self):
        """
        Get ensembled action from buffer.
        """
        with self._lock:
            return self._get_action_unlocked()

    def _get_action_unlocked(self):
        if self.timestep - self.actions_start_timestep >= len(self.actions):
            self.timestep += 1
            return None      # no data
        while self.actions_start_timestep < self.timestep:
            self.actions.popleft()
            self.actions_timestep.popleft()
            self.actions_start_timestep += 1
        actions = self.actions[0]
        actions_timestep = self.actions_timestep[0]
        if actions == []:
            self.timestep += 1
            return None      # no data
        # Sort by timestep only; default tuple sort would compare actions (arrays) on ties.
        sorted_actions = sorted(
            zip(actions_timestep, actions),
            key=lambda pair: np.asarray(pair[0]).item(),
        )
        all_actions = [x for _, x in sorted_actions]
        all_timesteps = [t for t, _ in sorted_actions]
        if self.max_past_inference_chunks is not None:
            max_chunks = self.max_past_inference_chunks + 1
            all_actions = all_actions[-max_chunks:]
            all_timesteps = all_timesteps[-max_chunks:]
        if self.mode == "new":
            action = all_actions[-1]
        elif self.mode == "old":
            action = all_actions[0]
        elif self.mode == "avg":
            action = np.array(all_actions).mean(axis = 0)
        elif self.mode == "act":
            weights = np.exp(-self.k * (self.timestep - np.array(all_timesteps)))
            weights = weights / weights.sum()
            action = (all_actions * weights.reshape((-1,) + (1,) * len(self.action_shape))).sum(axis = 0)
        elif self.mode == "hato":
            weights = self.tau ** (self.timestep - np.array(all_timesteps))
            weights = weights / weights.sum()
            action = (all_actions * weights.reshape((-1,) + (1,) * len(self.action_shape))).sum(axis = 0)
        else:
            raise AttributeError("Ensemble mode {} not supported.".format(self.mode))
        self.timestep += 1
        return action


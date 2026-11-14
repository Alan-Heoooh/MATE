import threading
import time
import numpy as np
import torch


class SharedVisionFeature:
    """
    Shared storage for latest vision feature and cloud.
    - VisionWorker updates the vision feature continuously
    - PolicyWorker gets the latest vision feature for inference
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.vision_feature = None  # torch.Tensor [feature_dim]
        self.cloud = None  # np.ndarray [N, 6] for visualization
        self.vision_obs_tcp_base = None  # np.ndarray [7], tcp pose at vision observation time in base frame
        self.timestamp = 0.0  # timestamp when feature was updated
        self.observation_time = 0.0  # timestamp when corresponding visual observation was captured
    

    def update(self, vision_feature: torch.Tensor, cloud: np.ndarray = None, observation_time: float = None, vision_obs_tcp_base: np.ndarray = None):
        """Update the latest vision feature and its observation timestamp."""
        with self.lock:
            self.vision_feature = vision_feature.detach().clone() if vision_feature is not None else None
            self.cloud = cloud.copy() if cloud is not None else None
            self.vision_obs_tcp_base = vision_obs_tcp_base.copy() if vision_obs_tcp_base is not None else None
            self.timestamp = time.time()
            self.observation_time = self.timestamp if observation_time is None else float(observation_time)


    def get_latest(self):
        """Get the latest vision feature, cloud, tcp pose at vision obs time, and observation timestamp."""
        with self.lock:
            return self.vision_feature.clone() if self.vision_feature is not None else None, \
                   self.cloud.copy() if self.cloud is not None else None, \
                   self.vision_obs_tcp_base.copy() if self.vision_obs_tcp_base is not None else None, \
                   self.observation_time
        

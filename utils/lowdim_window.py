"""
sample and project TCP force/torque and provides time windows based on freq/length
for the input of model when enable force_torque.input in data_cfg
"""
import time
import threading
from collections import deque
import numpy as np

from utils.transformation import xyz_rot_to_mat, mat_to_xyz_rot

class LowDimWindowProvider:
    """
    samples low dim data and provides time windows based on freq/length; projection to 'camera'/'base'/'tcp' coordinates is performed as needed.
    """
    def __init__(self, get_pose_fn, get_ft_fn, sample_hz: float = 100.0, buffer_secs: float = 60.0):
        self.get_pose_fn = get_pose_fn
        self.get_ft_fn = get_ft_fn
        self.sample_hz = float(sample_hz)
        self.dt = 1.0 / max(self.sample_hz, 1e-6)
        self._buf = deque(maxlen=int(self.sample_hz * buffer_secs))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, name="LowDimWindowProvider", daemon=True)
        self._th.start()

    def _loop(self):
        while not self._stop.is_set():
            tic = time.time()
            try:
                pose = self.get_pose_fn()
                T_base_tcp = xyz_rot_to_mat(pose, rotation_rep="quaternion")
                ft_tcp = self.get_ft_fn()
                with self._lock:
                    self._buf.append((tic, T_base_tcp.astype(np.float32), np.asarray(ft_tcp, dtype=np.float32)))
            except Exception:
                pass
            elapsed = time.time() - tic
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)

    def get_ft_window(self, *,
                   freq: float,
                   length: int,
                   coordinate: str,
                   projector,
                   remove_first: bool = True) -> np.ndarray:
        """
        Return (length, 6) float32 , transfer using projector.project_wrench()
          source='tcp', target=coordinate, tcp=T_base_tcp.
        """
        assert coordinate in ["camera", "base", "tcp"], f"coordinate must be one of ['camera', 'base', 'tcp'], got {coordinate}"
        assert freq <= self.sample_hz, "freq must be less than or equal to sample_hz"

        need = int(length) + (1 if remove_first else 0)
        dt = 1.0 / max(float(freq), 1e-6)

        with self._lock:
            if len(self._buf) == 0:
                return np.zeros((length, 6), dtype=np.float32)
            times = np.array([t for (t, _, __) in self._buf], dtype=np.float64)
            Ts = np.stack([T for (_, T, __) in self._buf], axis=0)
            fts = np.stack([ft for (_, __, ft) in self._buf], axis=0)

        t_end = times[-1]
        t_start = t_end - dt * need
        target_ts = np.linspace(t_start, t_end - dt, need, dtype=np.float64)

        idx = np.searchsorted(times, target_ts, side="left")
        idx = np.clip(idx, 0, len(times) - 1)

        sel_Ts = Ts[idx]
        sel_fts = fts[idx]

        proj_fts = projector.project_wrench(sel_fts, source_coord="tcp", target_coord=coordinate, tcp=sel_Ts)


        if remove_first and proj_fts.shape[0] > 0:
            proj_fts = proj_fts[1:]

        if proj_fts.shape[0] < length:
            pad = np.zeros((length - proj_fts.shape[0], 6), dtype=np.float32)
            proj_fts = np.concatenate([pad, proj_fts.astype(np.float32)], axis=0)

        return proj_fts.astype(np.float32)

    def get_tcp_window(self, *,
                   freq: float,
                   length: int,
                   coordinate: str,
                   projector,
                   remove_first: bool = True) -> np.ndarray:
        """
        Return (length, 7) float32 , transfer using projector.project_tcp()
        source_coord = "base", target_coord = coordinate
        """
        assert coordinate in ["camera", "base"]
        assert freq <= self.sample_hz, "freq must be less than or equal to sample_hz"
        need = int(length) + (1 if remove_first else 0)
        dt = 1.0 / max(float(freq), 1e-6)

        with self._lock:
            if len(self._buf) == 0:
                return np.zeros((length, 7), dtype=np.float32)
            times = np.array([t for (t, _, __) in self._buf], dtype=np.float64)
            Ts = np.stack([T for (_, T, __) in self._buf], axis=0)

        t_end = times[-1]
        t_start = t_end - dt * need
        target_ts = np.linspace(t_start, t_end - dt, need, dtype=np.float64)

        idx = np.searchsorted(times, target_ts, side="left")
        idx = np.clip(idx, 0, len(times) - 1)

        sel_Ts = Ts[idx]

        proj_tcps_mat = projector.project_tcp(
            tcp = sel_Ts,
            source_coord = "base",
            target_coord = coordinate,
            rotation_rep = "matrix"
        )
        proj_tcps = mat_to_xyz_rot(mat=proj_tcps_mat, rotation_rep="quaternion")

        if remove_first and proj_tcps.shape[0] > 0:
            proj_tcps = proj_tcps[1:]

        if proj_tcps.shape[0] < length:
            pad = np.zeros((length - proj_tcps.shape[0], 7), dtype=np.float32)
            proj_tcps = np.concatenate([pad, proj_tcps.astype(np.float32)], axis=0)

        return proj_tcps.astype(np.float32)

    def stop(self):
        self._stop.set()
        if self._th and self._th.is_alive():
            self._th.join()

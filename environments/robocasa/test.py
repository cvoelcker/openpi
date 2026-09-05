import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import robocasa  # noqa: F401 -- registers RoboCasa environments
import robosuite
from robosuite.controllers import load_composite_controller_config


config = {
    "env_name": "Kitchen",
    "robots": "PandaOmron",
    "controller_configs": load_composite_controller_config(robot="PandaOmron"),
    "has_renderer": False,
    "has_offscreen_renderer": True,
    "use_camera_obs": True,
    "camera_names": ["robot0_agentview_center"],
    "camera_heights": 128,
    "camera_widths": 128,
    "control_freq": 20,
    "ignore_done": True,
}

env = robosuite.make(**config)
obs = env.reset()
print("ENV_CLASS", type(env).__name__)
print("ACTION_DIM", env.action_dim)
for key, value in sorted(obs.items()):
    if hasattr(value, "shape"):
        print("OBS", key, tuple(value.shape), str(value.dtype))

action = np.zeros(env.action_dim, dtype=np.float32)
next_obs, reward, done, info = env.step(action)
print("STEP_OK", float(reward), bool(done), sorted(info.keys()))
image_keys = [key for key in next_obs if key.endswith("_image")]
print("HEADLESS_IMAGE_KEYS", image_keys)
assert image_keys, "Offscreen renderer did not return a camera image"
assert all(np.isfinite(value).all() for value in next_obs.values() if isinstance(value, np.ndarray))
env.close()

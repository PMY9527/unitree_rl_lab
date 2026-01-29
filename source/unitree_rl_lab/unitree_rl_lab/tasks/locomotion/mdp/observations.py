from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase

def cmg_q_ref(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return CMG reference joint positions. Shape: [num_envs, num_joints]."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmg_motion = env.extras.get("cmg_motion")
    if cmg_motion is None:
        # current joint positions before runner initializes CMG
        return asset.data.joint_pos[:, asset_cfg.joint_ids]
    return cmg_motion[:, :29]

def cmg_q_vel_ref(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return CMG reference joint velocities. Shape: [num_envs, num_joints]."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmg_motion = env.extras.get("cmg_motion")
    if cmg_motion is None:
        # current joint velocities before runner initializes CMG
        return asset.data.joint_vel[:, asset_cfg.joint_ids]
    return cmg_motion[:, 29:]
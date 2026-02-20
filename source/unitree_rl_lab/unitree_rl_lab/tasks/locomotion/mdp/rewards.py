from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


def joint_symmetry(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    pitch_pairs: list[list[str]],
    flip_pairs: list[list[str]],
) -> torch.Tensor:
    """Penalize left-right asymmetry, gated for straight walking (|vy| < 0.3, |yaw| < 0.3).

    pitch_pairs: joint pairs where q_left should equal q_right (pitch joints).
    flip_pairs: joint pairs where q_left should equal -q_right (roll/yaw joints).
    Returns mean squared error across all pairs (penalty, use negative weight).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Cache resolved joint indices
    if not hasattr(env, "_sym_cache"):
        env._sym_cache = {
            "pitch": [(asset.find_joints(p[0])[0], asset.find_joints(p[1])[0]) for p in pitch_pairs],
            "flip": [(asset.find_joints(p[0])[0], asset.find_joints(p[1])[0]) for p in flip_pairs],
        }
    # Gate: only enforce when going roughly straight
    cmd = env.command_manager.get_command(command_name)
    vy, yaw = cmd[:, 1], cmd[:, 2]
    gate = ((vy.abs() < 0.3) & (yaw.abs() < 0.3)).float()

    err = torch.zeros(env.num_envs, device=env.device)
    q = asset.data.joint_pos
    for l_idx, r_idx in env._sym_cache["pitch"]:
        err += (q[:, l_idx] - q[:, r_idx]).square().sum(dim=-1)
    for l_idx, r_idx in env._sym_cache["flip"]:
        err += (q[:, l_idx] + q[:, r_idx]).square().sum(dim=-1)
    n_pairs = len(pitch_pairs) + len(flip_pairs)
    return gate * err / max(n_pairs, 1)


def joint_pos_from_cmg_l2_gated(
    env, gated: bool, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of joint positions from CMG reference, activated via the commanded velocity, using exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    ref_motion = env.extras.get("cmg_motion")
    if ref_motion is None:
        # CMG motion not set (e.g., during play mode) - return zero reward
        return torch.zeros(env.num_envs, device=env.device)
    q_ref = ref_motion[:, :29]  # joint positions from CMG
    # exp(-0.6 * ||q - q_ref||^2)
    if gated:   
        cmd_vx = env.command_manager.get_command(command_name)[:, 0] # Commanded vx
        cmg_weight = torch.clamp((cmd_vx - 1.2) / 0.2, 0.0, 1.0) # Below 1.1 -> 0, 1.1 - 1.3 ramps up, 1.3 or above -> 1
        reward = cmg_weight * torch.exp(-0.6 * torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - q_ref), dim=1))
    else:
        reward = torch.exp(-0.6 * torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - q_ref), dim=1))
    return reward


def joint_vel_from_cmg_l2_gated(
    env, gated: bool, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of joint velocities from CMG reference, activated via the commanded velocity, using exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    ref_motion = env.extras.get("cmg_motion")
    if ref_motion is None:
        # CMG motion not set (e.g., during play mode) - return zero reward
        return torch.zeros(env.num_envs, device=env.device)
    qd_ref = ref_motion[:, 29:]  # joint velocities from CMG
    # exp(-0.5 * ||qd - qd_ref||^2)
    if gated:    
        cmd_vx = env.command_manager.get_command(command_name)[:, 0] # Commanded vx
        cmg_weight = torch.clamp((cmd_vx - 1.2) / 0.2, 0.0, 1.0) # Below 1.1 -> 0, 1.1 - 1.3 ramps up, 1.3 or above -> 1
        reward = cmg_weight * torch.exp(-0.5 * torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids] - qd_ref), dim=1))
    else:
        reward = torch.exp(-0.5 * torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids] - qd_ref), dim=1))
    return reward

def _walk_weight(env, command_name: str) -> torch.Tensor:
    """Compute walk weight: 1.0 when cmd_vx < 1.2, ramps to 0.0 at cmd_vx >= 1.3 (inverse of CMG gating)."""
    cmd_vx = env.command_manager.get_command(command_name)[:, 0]
    return torch.clamp(1.0 - (cmd_vx - 1.2) / 0.2, 0.0, 1.0)


def feet_gait_gated(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float = 0.5,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward *= cmd_norm > 0.1  # no gait reward when standing
    return _walk_weight(env, command_name) * reward


def feet_clearance_gated(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Foot clearance reward active only at low speeds (walk regime)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return _walk_weight(env, command_name) * torch.exp(-torch.sum(reward, dim=1) / std)


def base_height_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base height reward active only at low speeds (walk regime)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return _walk_weight(env, command_name) * reward


def undesired_contacts_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Undesired contacts penalty active only at low speeds (walk regime)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = torch.max(torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    reward = (net_forces > threshold).float()
    return _walk_weight(env, command_name) * reward


def lin_vel_z_l2_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize z-axis linear velocity, gated for low speeds only."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return _walk_weight(env, command_name) * torch.square(asset.data.root_lin_vel_b[:, 2])


def joint_vel_l2_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint velocities, gated for low speeds only."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _walk_weight(env, command_name) * torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def track_lin_vel_xy_yaw_frame_exp(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned
    robot frame using an exponential kernel.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), 
        dim=1
    )
    return torch.exp(-2.0 * lin_vel_error)

def track_ang_vel_z_world_exp(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error)

def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return 1.0 for terminated episodes (excluding timeouts)."""

    return (env.termination_manager.terminated & ~env.termination_manager.time_outs).float()


def action_magnitude_l2(env) -> torch.Tensor:
    """Penalize large action (residual) magnitudes to keep policy close to CMG reference."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


def action_smoothness_l2(env) -> torch.Tensor:
    """Penalize action jerk (second derivative) for smoother motion.

    Computes ||a_t - 2*a_{t-1} + a_{t-2}||^2 
    """
    a_t = env.action_manager.action
    a_t1 = env.action_manager.prev_action

    # Track a_{t-2} ourselves using env.extras
    if "_action_smoothness_prev_prev" not in env.extras:
        env.extras["_action_smoothness_prev_prev"] = a_t1.clone()
        return torch.zeros(env.num_envs, device=env.device)

    a_t2 = env.extras["_action_smoothness_prev_prev"]

    # Compute jerk
    jerk = a_t - 2 * a_t1 + a_t2

    # Update history for next step
    env.extras["_action_smoothness_prev_prev"] = a_t1.clone()

    return torch.sum(torch.square(jerk), dim=1)
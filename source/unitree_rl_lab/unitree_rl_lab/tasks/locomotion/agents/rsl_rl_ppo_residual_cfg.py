# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunnerResidual"
    num_steps_per_env = 24
    obs_groups = {"policy": ["policy"], "critic": ["policy", "gt_linear_velocity", "motion"]} 
    max_iterations = 15000
    save_interval = 500
    experiment_name = ""  # same as task name
    empirical_normalization = True
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[256],
        critic_hidden_dims=[256],
        activation="elu",
        rnn_type = "lstm",
        rnn_hidden_dim = 256,
        rnn_num_layers = 2,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005, # 0.005->0.001->0.0
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

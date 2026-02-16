// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "isaaclab/algorithms/algorithms.h"
#include "isaaclab/manager/observation_manager.h"
#include <mutex>
#include <atomic>

class State_RLResidual : public FSMState
{
public:
    State_RLResidual(int state_mode, std::string state_string);

    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();

        // Initialize final_action_for_control with default positions
        final_action_for_control.resize(env->robot->data.default_joint_pos.size());
        for (size_t i = 0; i < final_action_for_control.size(); ++i) {
            final_action_for_control[i] = env->robot->data.default_joint_pos[i];
        }

        // Start policy thread
        policy_thread_running = true;
        policy_initialized = false;
        policy_update_counter = 0;

        policy_thread = std::thread([this]{
            std::cout << "[Policy Thread] Started" << std::endl;
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            const float residual_clamp = 1.0f;
            const float residual_scale = 0.25f;

            int policy_debug_counter = 0;
            while (policy_thread_running)
            {
                bool print_debug = (policy_debug_counter % 50 == 0);

                env->robot->update();

                auto& joint_pos = env->robot->data.joint_pos;
                auto& joint_vel = env->robot->data.joint_vel;
                auto command = isaaclab::observations_map()["keyboard_velocity_commands"](env.get(), {});

                std::vector<float> joint_pos_vec(joint_pos.data(), joint_pos.data() + joint_pos.size());
                std::vector<float> joint_vel_vec(joint_vel.data(), joint_vel.data() + joint_vel.size());
                std::vector<float> command_vec(command.data(), command.data() + command.size());

                {
                    std::lock_guard<std::mutex> lock(action_mutex);

                    env->episode_length += 1;
                    env->robot->update();

                    cmg->forward_ar(joint_pos_vec, joint_vel_vec, command_vec);
                    auto qref = cmg->get_qref();

                    auto obs_map = env->observation_manager->compute();

                    auto residual = env->alg->act(obs_map);
                    env->action_manager->process_action(residual);
                    residual = env->action_manager->action();

                    float max_abs_residual = 0.0f;
                    for (size_t i = 0; i < residual.size(); ++i) {
                        float abs_r = std::abs(residual[i]);
                        if (abs_r > max_abs_residual) max_abs_residual = abs_r;
                        residual[i] = std::clamp(residual[i], -residual_clamp, residual_clamp);
                    }

                    final_action_for_control.resize(residual.size());
                    for (size_t i = 0; i < residual.size(); ++i) {
                        final_action_for_control[i] = qref[i] + residual_scale * residual[i];
                    }

                    // Store as last_action for next step
                    env->action_manager->process_action(final_action_for_control);

                    final_action_for_control[25] = 0.0f;  // left wrist pitch
                    final_action_for_control[26] = 0.0f;  // right wrist pitch

                    if (!policy_initialized) {
                        std::cout << "[Policy Thread] First action computed" << std::endl;
                        policy_initialized = true;
                    }
                    policy_update_counter++;
                }

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
                policy_debug_counter++;
            }
        });
    }

    void run();

    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::unique_ptr<isaaclab::CMGRunner> cmg;

    std::thread policy_thread;
    bool policy_thread_running = false;

    // Synchronization: track if policy has computed at least one action
    std::atomic<bool> policy_initialized{false};
    std::atomic<int> policy_update_counter{0};

    // Mutex to protect action/qref reads and writes
    std::mutex action_mutex;
    
    std::vector<float> final_action_for_control;
};

REGISTER_FSM(State_RLResidual)

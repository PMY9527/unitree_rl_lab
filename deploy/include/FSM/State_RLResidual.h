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

        std::cout << "[State_RLResidual] Entering state - starting policy thread" << std::endl;

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

            int policy_debug_counter = 0;
            while (policy_thread_running)
            {
                bool print_policy_debug = (policy_debug_counter % 100 == 0);

                // CRITICAL: Update robot state BEFORE using it for CMG
                // This ensures CMG and policy see the same robot state
                env->robot->update();

                // Get current robot state
                auto& joint_pos = env->robot->data.joint_pos;
                auto& joint_vel = env->robot->data.joint_vel;

                // Get velocity command from registered observation
                auto command = isaaclab::observations_map()["keyboard_velocity_commands"](env.get(), {});

                if (print_policy_debug) {
                    std::cout << "\n[RLRESIDUAL POLICY THREAD] ========== Step " << policy_debug_counter << " ==========" << std::endl;

                    // Print raw robot state
                    std::cout << "[RLRESIDUAL POLICY] Raw joint pos (first 5): [";
                    for (int i = 0; i < std::min(5, (int)joint_pos.size()); ++i) {
                        std::cout << joint_pos[i] << (i < 4 ? ", " : "");
                    }
                    std::cout << "...]" << std::endl;

                    std::cout << "[RLRESIDUAL POLICY] Raw joint vel (first 5): [";
                    for (int i = 0; i < std::min(5, (int)joint_vel.size()); ++i) {
                        std::cout << joint_vel[i] << (i < 4 ? ", " : "");
                    }
                    std::cout << "...]" << std::endl;

                    // Print observations that will be fed to policy
                    auto obs_joint_pos_rel = isaaclab::observations_map()["joint_pos_rel"](env.get(), {});
                    std::cout << "[RLRESIDUAL POLICY] Observation joint_pos_rel (first 5): [";
                    for (int i = 0; i < std::min(5, (int)obs_joint_pos_rel.size()); ++i) {
                        std::cout << obs_joint_pos_rel[i] << (i < 4 ? ", " : "");
                    }
                    std::cout << "...]" << std::endl;

                    auto obs_joint_vel_rel = isaaclab::observations_map()["joint_vel_rel"](env.get(), {});
                    std::cout << "[RLRESIDUAL POLICY] Observation joint_vel_rel (first 5): [";
                    for (int i = 0; i < std::min(5, (int)obs_joint_vel_rel.size()); ++i) {
                        std::cout << obs_joint_vel_rel[i] << (i < 4 ? ", " : "");
                    }
                    std::cout << "...]" << std::endl;

                    auto obs_last_action = isaaclab::observations_map()["last_action"](env.get(), {});
                    std::cout << "[RLRESIDUAL POLICY] Observation last_action (first 5): [";
                    for (int i = 0; i < std::min(5, (int)obs_last_action.size()); ++i) {
                        std::cout << obs_last_action[i] << (i < 4 ? ", " : "");
                    }
                    std::cout << "...]" << std::endl;

                    std::cout << "[RLRESIDUAL POLICY] Observation velocity command: [" << command[0] << ", " << command[1] << ", " << command[2] << "]" << std::endl;
                }

                // CMG takes CURRENT robot state as input (not autoregressive!)
                std::vector<float> joint_pos_vec(joint_pos.data(), joint_pos.data() + joint_pos.size());
                std::vector<float> joint_vel_vec(joint_vel.data(), joint_vel.data() + joint_vel.size());
                std::vector<float> command_vec(command.data(), command.data() + command.size());

                // Run CMG forward pass with current robot state
                cmg->forward(joint_pos_vec, joint_vel_vec, command_vec);

                // CRITICAL: Lock the entire sequence to prevent control thread from reading
                // intermediate state (residual only) instead of final action (qref + residual)
                {
                    std::lock_guard<std::mutex> lock(action_mutex);

                    // Run RL policy to get residual
                    env->step();

                    // Get RAW residual (before scale/offset/clip processing)
                    auto residual = env->action_manager->action();
                    auto qref = cmg->get_qref();

                    if (print_policy_debug) {
                        std::cout << "[RLRESIDUAL POLICY] CMG qref (first 5): [";
                        for (int i = 0; i < std::min(5, (int)qref.size()); ++i) {
                            std::cout << qref[i] << (i < 4 ? ", " : "");
                        }
                        std::cout << "...]" << std::endl;

                        std::cout << "[RLRESIDUAL POLICY] RL residual (first 5): [";
                        for (int i = 0; i < std::min(5, (int)residual.size()); ++i) {
                            std::cout << residual[i] << (i < 4 ? ", " : "");
                        }
                        std::cout << "...]" << std::endl;
                    }

                    // TEST: Send only residual to motors (no CMG qref)
                    // This matches RLBase behavior to test if CMG is causing instability
                    final_action_for_control.resize(residual.size());
                    for (size_t i = 0; i < residual.size(); ++i) {
                        final_action_for_control[i] = residual[i];  // ONLY residual, no qref!
                        // final_action_for_control[i] = qref[i] + residual[i];  // Original: qref + residual
                    }

                    if (print_policy_debug) {
                        std::cout << "[RLRESIDUAL POLICY] Final action (residual only, NO CMG): [";
                        for (int i = 0; i < std::min(5, (int)final_action_for_control.size()); ++i) {
                            std::cout << final_action_for_control[i] << (i < 4 ? ", " : "");
                        }
                        std::cout << "...]" << std::endl;
                    }

                    // CRITICAL: Do NOT call process_action(final_action)!
                    // Keep residual in action_manager so next iteration's last_action observation
                    // reads the residual (matching RLBase behavior, where it works stably)
                    // The final_action_for_control is stored separately for control thread

                    if (!policy_initialized) {
                        std::cout << "[Policy Thread] First action computed, control thread can now proceed" << std::endl;
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

    // Store final action (qref + residual) for control thread
    // action_manager stores only residual for last_action observation
    std::vector<float> final_action_for_control;
};

REGISTER_FSM(State_RLResidual)

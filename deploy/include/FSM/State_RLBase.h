// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
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

        std::cout << "[State_RLBase] Entering state - starting policy thread" << std::endl;

        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            std::cout << "[RLBase Policy Thread] Started" << std::endl;
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

                if (print_policy_debug) {
                    std::cout << "\n[RLBASE POLICY THREAD] ========== Step " << policy_debug_counter << " ==========" << std::endl;

                    // Print raw robot state
                    std::cout << "[RLBASE POLICY] Raw joint pos (first 5): [";
                    for (int i = 0; i < std::min(5, (int)env->robot->data.joint_pos.size()); ++i) {
                        std::cout << env->robot->data.joint_pos[i] << (i < 4 ? ", " : "");
                    }
                    std::cout << "...]" << std::endl;

                    std::cout << "[RLBASE POLICY] Raw joint vel (first 5): [";
                    for (int i = 0; i < std::min(5, (int)env->robot->data.joint_vel.size()); ++i) {
                        std::cout << env->robot->data.joint_vel[i] << (i < 4 ? ", " : "");
                    }
                    std::cout << "...]" << std::endl;

                    // Print observations that will be fed to policy
                    auto obs_joint_pos_rel = isaaclab::observations_map()["joint_pos_rel"](env.get(), {});
                    std::cout << "[RLBASE POLICY] Observation joint_pos_rel (first 5): [";
                    for (int i = 0; i < std::min(5, (int)obs_joint_pos_rel.size()); ++i) {
                        std::cout << obs_joint_pos_rel[i] << (i < 4 ? ", " : "");
                    }
                    std::cout << "...]" << std::endl;

                    auto obs_joint_vel_rel = isaaclab::observations_map()["joint_vel_rel"](env.get(), {});
                    std::cout << "[RLBASE POLICY] Observation joint_vel_rel (first 5): [";
                    for (int i = 0; i < std::min(5, (int)obs_joint_vel_rel.size()); ++i) {
                        std::cout << obs_joint_vel_rel[i] << (i < 4 ? ", " : "");
                    }
                    std::cout << "...]" << std::endl;

                    auto obs_last_action = isaaclab::observations_map()["last_action"](env.get(), {});
                    std::cout << "[RLBASE POLICY] Observation last_action (first 5): [";
                    for (int i = 0; i < std::min(5, (int)obs_last_action.size()); ++i) {
                        std::cout << obs_last_action[i] << (i < 4 ? ", " : "");
                    }
                    std::cout << "...]" << std::endl;

                    auto command = isaaclab::observations_map()["keyboard_velocity_commands"](env.get(), {});
                    std::cout << "[RLBASE POLICY] Observation velocity command: [" << command[0] << ", " << command[1] << ", " << command[2] << "]" << std::endl;
                }

                env->step();

                if (print_policy_debug) {
                    auto action = env->action_manager->processed_actions();
                    std::cout << "[RLBASE POLICY] Output action (first 5): [";
                    for (int i = 0; i < std::min(5, (int)action.size()); ++i) {
                        std::cout << action[i] << (i < 4 ? ", " : "");
                    }
                    std::cout << "...]" << std::endl;
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

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)

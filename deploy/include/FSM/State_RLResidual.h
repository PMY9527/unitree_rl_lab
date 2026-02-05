// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "isaaclab/algorithms/algorithms.h"
#include "isaaclab/manager/observation_manager.h"

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
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                // Get robot state for CMG
                auto& joint_pos = env->robot->data.joint_pos;
                auto& joint_vel = env->robot->data.joint_vel;

                // Get velocity command from registered observation
                auto command = isaaclab::observations_map()["keyboard_velocity_commands"](env.get(), {});

                // Run CMG forward pass
                cmg->forward(joint_pos, joint_vel, command);

                // Run RL policy
                env->step();

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
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
};

REGISTER_FSM(State_RLResidual)

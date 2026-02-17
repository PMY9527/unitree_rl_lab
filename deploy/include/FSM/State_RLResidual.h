// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "isaaclab/algorithms/algorithms.h"

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
                env->step();

                // CMG runs after policy step
                auto& jp = env->robot->data.joint_pos;
                auto& jv = env->robot->data.joint_vel;
                auto cmd = isaaclab::observations_map()["keyboard_velocity_commands"](env.get(), {});
                cmg->forward( // forward_ar() for AR
                    {jp.data(), jp.data() + jp.size()},
                    {jv.data(), jv.data() + jv.size()},
                    {cmd.data(), cmd.data() + cmd.size()}
                );
                auto qr = cmg->get_qref();
                for (size_t i = 0; i < qref.size(); ++i)
                    qref[i] = qr[i];

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
    std::vector<float> qref, final_action;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLResidual)

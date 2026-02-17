#include "FSM/State_RLResidual.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_RLResidual::State_RLResidual(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    auto cmg_dir = param::parser_policy_dir(cfg["cmg_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");
    cmg = std::make_unique<isaaclab::CMGRunner>(cmg_dir / "exported" / "cmg_exported.onnx", cmg_dir / "stats" / "cmg_stats.yaml");

    // Pre-allocate qref
    qref.resize(env->robot->data.default_joint_pos.size(), 0.0f);
    final_action.resize(env->robot->data.default_joint_pos.size(), 0.0f);

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLResidual::run()
{
    auto residual = env->action_manager->processed_actions();
                   
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        final_action[i] = qref[i] + residual[i];
        final_action[25] = 0.0f;
        final_action[26] = 0.0f;
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = final_action[i];
    }
}

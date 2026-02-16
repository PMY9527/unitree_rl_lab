#include "FSM/State_RLResidual.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {0.2f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        // TODO: smooth and limit the velocity commands
        cmd = key_commands[key];
    }
    return cmd;
}

}

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
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy_mlp.onnx");
    cmg = std::make_unique<isaaclab::CMGRunner>(cmg_dir / "exported" / "cmg_exported.onnx", cmg_dir / "stats" / "cmg_stats.yaml");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLResidual::run()
{
    static int debug_counter = 0;
    static int total_calls = 0;
    static bool warned_not_initialized = false;
    bool print_debug = false; // (debug_counter % 100 == 0);  // Print every 100 steps

    total_calls++;

    // Wait for policy thread to compute first action before sending commands
    if (!policy_initialized) {
        if (!warned_not_initialized) {
            std::cout << "[State_RLResidual] Waiting for policy to initialize..." << std::endl;
            warned_not_initialized = true;
        }
        // Keep robot at default positions until policy is ready
        for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
            lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = env->robot->data.default_joint_pos[i];
        }
        return;
    }

    // Lock while reading final action (qref + residual stored separately from action_manager)
    std::vector<float> action;
    {
        std::lock_guard<std::mutex> lock(action_mutex);
        action = final_action_for_control;
    }

    if (print_debug) {
        std::cout << "\n[RLRESIDUAL DEBUG] ========== Step " << debug_counter << " (Policy updates: " << policy_update_counter.load() << ") ==========" << std::endl;
        std::cout << "[RLRESIDUAL] Current joint pos (first 5): [";
        for (int i = 0; i < std::min(5, (int)env->robot->data.joint_pos.size()); ++i) {
            std::cout << env->robot->data.joint_pos[i] << (i < 4 ? ", " : "");
        }
        std::cout << "...]" << std::endl;

        std::cout << "[RLRESIDUAL] Current joint vel (first 5): [";
        for (int i = 0; i < std::min(5, (int)env->robot->data.joint_vel.size()); ++i) {
            std::cout << env->robot->data.joint_vel[i] << (i < 4 ? ", " : "");
        }
        std::cout << "...]" << std::endl;

        std::cout << "[RLRESIDUAL] Final action (qref+residual) (first 5): [";
        for (int i = 0; i < std::min(5, (int)action.size()); ++i) {
            std::cout << action[i] << (i < 4 ? ", " : "");
        }
        std::cout << "...]" << std::endl;

        // Check position delta from current
        float max_delta_current = 0.0f;
        for (int i = 0; i < std::min((int)action.size(), (int)env->robot->data.joint_pos.size()); ++i) {
            float delta = std::abs(action[i] - env->robot->data.joint_pos[i]);
            if (delta > max_delta_current) max_delta_current = delta;
        }
        std::cout << "[RLRESIDUAL] Max position delta from current: " << max_delta_current << " rad" << std::endl;
    }

    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }

    if (print_debug) {
        std::cout << "[RLRESIDUAL] Final command (first 5): [";
        for (int i = 0; i < std::min(5, (int)env->robot->data.joint_ids_map.size()); ++i) {
            std::cout << lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() << (i < 4 ? ", " : "");
        }
        std::cout << "...]" << std::endl;
    }

    debug_counter++;
}
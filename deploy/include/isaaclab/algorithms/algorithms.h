// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

// LSTM model has:                                                                                                                                      
//   - Inputs: obs (480), h_in (2×1×256), c_in (2×1×256)                                                                                                                
//   - Outputs: actions (29), h_out (2×1×256), c_out (2×1×256) 
// TODO: import CMG. 
#pragma once

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <mutex>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.release());
        }

        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                size *= dim;
            }
            input_sizes.push_back(size);
        }

        // Get all output names and shapes
        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            Ort::TypeInfo output_type = session->GetOutputTypeInfo(i);
            output_shapes.push_back(output_type.GetTensorTypeAndShapeInfo().GetShape());
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            output_names.push_back(output_name.release());
        }

        // First output is actions
        action.resize(output_shapes[0][1]);

        // Initialize hidden states for LSTM (h_in, c_in)
        for (size_t i = 0; i < input_names.size(); ++i) {
            std::string name_str(input_names[i]);
            if (name_str == "h_in" || name_str == "c_in") {
                hidden_states[name_str] = std::vector<float>(input_sizes[i], 0.0f);
                std::cout << "[OrtRunner] Initialized " << name_str << " with size " << input_sizes[i] << std::endl;
            }
        }
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for(int i(0); i<input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);

            // Use hidden states for h_in/c_in, observations for everything else
            std::vector<float>* input_data_ptr;
            if (hidden_states.find(name_str) != hidden_states.end()) {
                input_data_ptr = &hidden_states[name_str];
            } else {
                if (obs.find(name_str) == obs.end()) {
                    throw std::runtime_error("Input name " + name_str + " not found in observations.");
                }
                input_data_ptr = &obs.at(name_str);
            }

            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data_ptr->data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());

        // Copy action output (first output)
        auto action_data = output_tensors[0].GetTensorMutableData<float>();
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), action_data, output_shapes[0][1] * sizeof(float));

        // Update hidden states (h_out -> h_in, c_out -> c_in)
        for (size_t i = 0; i < output_names.size(); ++i) {
            std::string output_name_str(output_names[i]);
            if (output_name_str == "h_out") {
                auto h_out_data = output_tensors[i].GetTensorMutableData<float>();
                size_t h_size = 1;
                for (auto dim : output_shapes[i]) h_size *= dim;
                std::memcpy(hidden_states["h_in"].data(), h_out_data, h_size * sizeof(float));
            } else if (output_name_str == "c_out") {
                auto c_out_data = output_tensors[i].GetTensorMutableData<float>();
                size_t c_size = 1;
                for (auto dim : output_shapes[i]) c_size *= dim;
                std::memcpy(hidden_states["c_in"].data(), c_out_data, c_size * sizeof(float));
            }
        }

        return action;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<int64_t> input_sizes;

    std::unordered_map<std::string, std::vector<float>> hidden_states;
};

// CMG Runner: Controlled Motion Generator
// Input: joint_pos[29] + joint_vel[29] + command[3]
// Output: motion_ref[58] (reference joint positions + velocities)
class CMGRunner
{
public:
    CMGRunner(const std::string& model_path, const std::string& data_path)
    {
        // Init ONNX Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "cmg_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        // Get input info
        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names_owned.push_back(std::move(input_name));
            input_names.push_back(input_names_owned.back().get());

            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
        }

        // Get output info
        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            output_names_owned.push_back(std::move(output_name));
            output_names.push_back(output_names_owned.back().get());

            Ort::TypeInfo output_type = session->GetOutputTypeInfo(i);
            output_shapes.push_back(output_type.GetTensorTypeAndShapeInfo().GetShape());
        }

        // Load normalization stats
        load_stats(data_path);

        // Pre-allocate buffers
        motion_ref.resize(motion_dim);
        motion_norm.resize(motion_dim);
        cmd_norm.resize(cmd_dim);

        std::cout << "[CMGRunner] Loaded model: " << model_path << std::endl;
        std::cout << "[CMGRunner] motion_dim=" << motion_dim << ", cmd_dim=" << cmd_dim << std::endl;
    }

    void load_stats(const std::string& data_path)
    {
        YAML::Node data = YAML::LoadFile(data_path);
        YAML::Node stats = data["stats"];

        motion_mean = stats["motion_mean"].as<std::vector<float>>();
        motion_std = stats["motion_std"].as<std::vector<float>>();
        cmd_min = stats["command_min"].as<std::vector<float>>();
        cmd_max = stats["command_max"].as<std::vector<float>>();

        motion_dim = motion_mean.size();  // 58
        cmd_dim = cmd_min.size();         // 3

        std::cout << "[CMGRunner] Loaded stats from: " << data_path << std::endl;
    }

    // Main forward pass
    // Input: joint_pos[29], joint_vel[29], command[3]
    // Output: motion_ref[58] = [pos_ref(29), vel_ref(29)]
    std::vector<float> forward(const std::vector<float>& joint_pos,
                                const std::vector<float>& joint_vel,
                                const std::vector<float>& command)
    {
        // 1. Normalize motion: concat(joint_pos, joint_vel), then normalize
        size_t pos_size = joint_pos.size();
        for (size_t i = 0; i < pos_size; ++i) {
            motion_norm[i] = (joint_pos[i] - motion_mean[i]) / motion_std[i];
        }
        for (size_t i = 0; i < joint_vel.size(); ++i) {
            motion_norm[pos_size + i] = (joint_vel[i] - motion_mean[pos_size + i]) / motion_std[pos_size + i];
        }

        // 2. Normalize command: (cmd - min) / (max - min) * 2 - 1 -> [-1, 1]
        for (size_t i = 0; i < cmd_dim; ++i) {
            float range = cmd_max[i] - cmd_min[i];
            if (range > 1e-6f) {
                cmd_norm[i] = (command[i] - cmd_min[i]) / range * 2.0f - 1.0f;
            } else {
                cmd_norm[i] = 0.0f;
            }
        }

        // 3. Run ONNX inference
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, motion_norm.data(), motion_dim,
            input_shapes[0].data(), input_shapes[0].size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, cmd_norm.data(), cmd_dim,
            input_shapes[1].data(), input_shapes[1].size()));

        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), input_tensors.size(),
            output_names.data(), output_names.size());

        // 4. Denormalize output: output * std + mean
        auto* output_data = output_tensors[0].GetTensorMutableData<float>();
        std::lock_guard<std::mutex> lock(mtx_);
        for (size_t i = 0; i < motion_dim; ++i) {
            motion_ref[i] = output_data[i] * motion_std[i] + motion_mean[i];
        }

        return motion_ref;
    }

    // Get cached motion_ref (thread-safe)
    std::vector<float> get_motion_ref()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return motion_ref;
    }

    // Get only position reference [0:29]
    std::vector<float> get_qref()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return std::vector<float>(motion_ref.begin(), motion_ref.begin() + motion_dim / 2);
    }

private:
    // ONNX Runtime
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    // Names
    std::vector<Ort::AllocatedStringPtr> input_names_owned;
    std::vector<Ort::AllocatedStringPtr> output_names_owned;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;

    // Normalization stats
    std::vector<float> motion_mean, motion_std;
    std::vector<float> cmd_min, cmd_max;
    size_t motion_dim = 58;
    size_t cmd_dim = 3;

    // Buffers
    std::vector<float> motion_ref;        // output [58]
    std::vector<float> motion_norm; // normalized input
    std::vector<float> cmd_norm;    // normalized command

    std::mutex mtx_;
};

};